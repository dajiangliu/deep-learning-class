Sys.setenv(LD_LIBRARY_PATH=paste("/usr/local/cuda-11.0/targets/x86_64-linux/lib/", Sys.getenv("LD_LIBRARY_PATH"),sep=":",collapse=":"))
library(keras)
library(tfdatasets)
library(tensorflow);
library(reticulate);

## have to load .so files from different versions of cuda; blame on R tensorflow; 
dyn.load('/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudart.so.11.0');
dyn.load('/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcublas.so.11');
dyn.load('/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcublasLt.so.11');
dyn.load('/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcusparse.so.11');
dyn.load('/usr/local/cuda/lib64/libcudnn.so.8');
##import_from_path('scipy',path = "~/.local/lib/python3.6/site-packages/");
## a AlexNet implementation;
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
    layer_dense(units = 10, activation = "softmax");

model %>% compile(
              optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = "accuracy"
          )
##this is the precompiled version cifar10 data; 
dat <- readRDS('cifar227x227.rds');
history <- model %>% 
    fit(
        x = dat$train$x, y = dat$train$y,
        epochs = 10,
        validation_data = unname(dat$test),
        verbose = 2
    )
