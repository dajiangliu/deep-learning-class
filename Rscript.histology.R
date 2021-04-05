Sys.setenv(LD_LIBRARY_PATH=paste("/usr/local/cuda-11.0/targets/x86_64-linux/lib/", Sys.getenv("LD_LIBRARY_PATH"),sep=":",collapse=":"))
library(keras);
library(tfdatasets);
library(tensorflow);
dyn.load('/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudart.so.11.0');
dyn.load('/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcublas.so.11');
dyn.load('/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcublasLt.so.11');
dyn.load('/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusparse.so.11');
dyn.load('/usr/local/cuda/lib64/libcudnn.so.8');

data.dir <- "Kather_texture_2016_image_tiles_5000";
images <- list.files(data.dir, pattern = ".jpg", recursive = TRUE)
##length(images)
classes <- list.dirs(data.dir, full.names = FALSE, recursive = FALSE)
##set up the iterator to loop through the images in the data; 
list_ds <- file_list_dataset(file_pattern = paste0(data.dir, "/*/*"))
list_ds %>% reticulate::as_iterator() %>% reticulate::iter_next()

##based upon the files and folders, need to pair the dataset and the labels; 
get_label <- function(file_path) {
  parts <- tf$strings$split(file_path, "/")
  parts[-2] %>% 
    tf$equal(classes) %>% 
    tf$cast(dtype = tf$float64)
}

decode_img <- function(file_path, height = 227, width = 227) {
  
  size <- as.integer(c(height, width))
  
  file_path %>% 
    tf$io$read_file() %>% 
    tf$image$decode_jpeg(channels = 3) %>% 
    tf$image$convert_image_dtype(dtype = tf$float64) %>% 
    tf$image$resize(size = size)
}

preprocess_path <- function(file_path) {
  list(
    decode_img(file_path),
    get_label(file_path)
  )
}

labeled_ds <- list_ds %>% 
    dataset_map(.,preprocess_path)

labeled_ds %>% 
  reticulate::as_iterator() %>% 
  reticulate::iter_next()

prepare <- function(ds, batch_size, shuffle_buffer_size) {
  
  if (shuffle_buffer_size > 0)
    ds <- ds %>% dataset_shuffle(shuffle_buffer_size)
  
  ds %>% 
    dataset_batch(batch_size) %>% 
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    dataset_prefetch(buffer_size = tf$data$experimental$AUTOTUNE)
}

model <- keras_model_sequential() %>%
    layer_conv_2d(filters = 96, kernel_size = c(11,11), activation = "relu", 
                  input_shape = c(227,227,3),stride=4) %>%
    layer_max_pooling_2d(pool_size = c(3,3),stride=2) %>% 
    layer_conv_2d(filters = 96, kernel_size = c(5,5), activation = "relu", 
                  input_shape = c(27,27,96),stride=1) %>%
    layer_max_pooling_2d(pool_size = c(3,3),stride=2) %>% 
    layer_conv_2d(filters = 384, kernel_size = c(3,3), activation = "relu", 
                  input_shape = c(13,13,384),stride=1) %>%
    layer_conv_2d(filters = 384, kernel_size = c(3,3), activation = "relu", 
                  input_shape = c(13,13,384),stride=1) %>%
    layer_conv_2d(filters = 256, kernel_size = c(3,3), activation = "relu", 
                  input_shape = c(13,13,384),stride=1) %>%
    layer_max_pooling_2d(pool_size = c(3,3),stride=2) %>%
    layer_flatten() %>% 
    layer_dense(units = 9216, activation = "relu") %>% 
    layer_dense(units = 4096, activation = "relu") %>%
    layer_dense(units = 4096, activation = "relu") %>%
    layer_dense(units = 8, activation = "softmax");

model %>% compile(
              optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = "accuracy"
          )
model %>% 
  fit(
    prepare(labeled_ds, batch_size = 32, shuffle_buffer_size = 1000),
    epochs = 5,
    verbose = 2
  )

