#####################
### MovieLens Project
###
### JV Tagaro
#####################
# Parameters And Coefficients
movie_k <- 2        # Movie Regularization Parameter
genre_k <- 1        # User-Genre Regularization Parameter
################################################################################
#  #
#####################
### Libraries
#####################
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))     install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(gridExtra)
library(lubridate)
#####################
### Provided Code
#####################
options(timeout = 120)

dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)

movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)

ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))

movielens <- left_join(ratings, movies, by = "movieId")

# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# set.seed(1) # if using R 3.5 or earlier
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)

# rm unused variables
rm(dl, ratings, movies, test_index, temp, movielens, removed)
#####################
### Get Start Time
#####################
current_time <- now()
#####################
### Data Summary
#####################
# Get average, minimum and maximum ratings.
overall_average <- edx %>% 
  summarise(ave_rat = mean(rating),
            min_rat = min(rating),
            max_rat = max(rating) )
max_rating <- overall_average$max_rat
min_rating <- overall_average$min_rat
average_rating <- overall_average$ave_rat
#####################
### Get Genre Details
#####################
# Get Unique Genres
genres_in_data <- edx %>% 
  mutate(genreList = str_extract_all(genres, "[a-zA-Z\\-\\s]+")) %>% 
  .$genreList %>% 
  unlist() %>% 
  unique()
# Total Number of Genres
total_genres <- length(genres_in_data)
# Function to generate regex patterns for each genre.
get_pattern <- function(genre){
  # pattern should detect specified genre regardless of its position
  #   in the string, the 'charCombo' variable is used to handle all
  #   characters of the 'genre' string that is not the specified genre.
  charCombo <- "[a-zA-Z\\| ]*"
  # Sandwich specified genre in between 2 'charCombo' so that the
  #   pattern can detect the genre regardless of position.
  pattern <- str_c( c(charCombo, genre, charCombo), collapse = "")
  pattern
}
# Generate patterns for each genre in genres_in_data
genres_with_pattern <- genres_in_data %>% 
  as.data.frame() %>% 
  # put data in 'genres' column then remove '.' column
  mutate(genres = .[]) %>%  
  select(genres) %>%
  # use rowwise to get pattern then un-group afterwards 
  rowwise(genres) %>% 
  mutate(genre_pattern = get_pattern(genres)) %>% 
  ungroup()
##################
### Get Regularized Movie Bias
##################
# movie_k parameter used to minimize error for low total.
movie_residuals_regularized <- edx %>% 
  group_by(movieId) %>% 
  summarise(total = n(),
            divisor = total + movie_k,
            movie_residual = sum(rating - average_rating)/divisor) 
# Mean of Movie Bias Used to replace NAs if there is.
mean_movie <- movie_residuals_regularized %>% 
  summarise(ave = mean(movie_residual)) %>% 
  .$ave
##################
### Get Regularized User-Genre Bias 
##################
# Function for getting genre residual of each user.
get_genre_residual_regularized <- function(i){
  # Get a row from 'genres_with_pattern' for the genre and the pattern
  gen <- genres_with_pattern$genres[i]
  pat <- genres_with_pattern$genre_pattern[i]
  # use the pattern to get the residual of user for this genre
  # genre_k parameter used to minimize error for low total
  edx %>% 
    filter(str_detect(genres,pat)) %>% 
    group_by(userId) %>% 
    summarise("{gen}_total":= n(),
              "{gen}_residual":= (sum(rating-average_rating)/(n() + genre_k))
    )
}
# Apply the function to each genre using lapply
genre_unprocessed <- lapply(1:total_genres, get_genre_residual_regularized)
# : Result is a list of data frames corresponding to each genre 
#   that contains each user's residual.
#
# Use 'reduce' to left_join all resulting genre
#   then use replace() to remove NAs or genres the user/s did not rate yet.
genre_residual_regularized <- reduce(genre_unprocessed,left_join,by="userId") %>% 
  replace(is.na(.),0)
#################
### Final evaluation
##################
# Use the final model formula in the report for prediction:
# Y = (0.9 * x_ug) + (0.8 * x_m) 
# Use final_holdout_test predict ratings
new_res <- final_holdout_test %>% 
  # Join User-Genre Bias
  left_join(genre_residual_regularized, by = "userId") %>% 
  # For additional measure, use replace_na to remove NAs
  mutate(across(everything(),~replace_na(.x,0))) %>% 
  # Transform data into long data, for easier calculations, residual value is b_gui in formula
  pivot_longer(cols = ends_with("residual"),
               names_to = "residual_type",
               values_to = "residual_value",
               names_pattern = "^([a-zA-Z-\\s]*)[a-zA-Z_]*_residual") %>% 
  select(userId,movieId,rating,title,genres,residual_type,residual_value) %>% 
  # Detect if movie has the genre specified in 'residual_type' : g_mi in formula
  mutate(residual_multiplier = ifelse(str_detect(residual_type,genres),1,0)) %>% 
  # transform residual value using multiplier
  mutate(residual_value = residual_value * residual_multiplier) %>%
  # Undo transform while using 'summarise' to get User-Genre Bias : x_ug
  group_by(userId,movieId) %>% 
  summarise(genre_residual = sum(residual_value)/(sum(residual_multiplier)),
            rating = first(rating),          # Rating is constant if same userId & movieId
            title = first(title),            # preserve title
            .groups = "drop") %>% 
  # Join Regularized Movie Bias.
  left_join(movie_residuals_regularized %>% select(movieId, movie_residual), by = "movieId") %>% 
  # Remove NAs, if there is.
  mutate(movie_residual = ifelse(is.na(movie_residual), mean_movie, movie_residual)) %>% 
  # For NAs that might be introduced by some error in calculations.
  replace(is.na(.),0) %>% 
  # Using the formula, get the predicted ratings
  mutate(predicted_rating = (0.9 * genre_residual) + (0.8 * movie_residual) + average_rating) %>% 
  # Adjustments
  mutate(predicted_rating = ifelse(predicted_rating > max_rating,
                                   max_rating, predicted_rating)) %>% 
  mutate(predicted_rating = ifelse(predicted_rating < min_rating,
                                   min_rating, predicted_rating))
# Time at End of prediction
time_elapsed <- difftime(now(),current_time) %>% round(2)
time_elapsed
##################
# Show FINAL RMSE
##################
RMSE(new_res$predicted_rating,new_res$rating) %>%  knitr::kable(col.names = "RMSE")
