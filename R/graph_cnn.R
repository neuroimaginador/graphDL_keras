GraphConvolutionLayer <- R6::R6Class("GraphConvolutionLayer",
                                     
                                     inherit = KerasLayer,
                                     
                                     public = list(
                                       
                                       num_filters = NULL,
                                       
                                       adj = NULL,
                                       
                                       activation = NULL,
                                       
                                       dropout = NULL,
                                       
                                       weights = NULL,
                                       
                                       bias = NULL,
                                       
                                       use_bias = NULL,
                                       
                                       initialize = function(num_filters, 
                                                             adj,
                                                             dropout = 0, 
                                                             use_bias = TRUE,
                                                             activation = activation_relu) {
                                         
                                         self$num_filters <- num_filters
                                         self$adj <- adj #k_permute_dimensions(adj, c(1, 3, 2))
                                         self$dropout <- dropout
                                         self$activation <- activation
                                         self$use_bias <- use_bias
                                         
                                       },
                                       
                                       build = function(input_shape) {
                                         
                                         self$weights <- self$add_weight(
                                           name = 'weights', 
                                           shape = list(input_shape[[3]],
                                                        self$num_filters),
                                           initializer = initializer_random_normal(),
                                           trainable = TRUE
                                         )
                                         
                                         if (self$use_bias) {
                                           
                                           self$bias <- self$add_weight(
                                             name = "bias",
                                             shape = list(input_shape[[2]], 
                                                          self$num_filters),
                                             initializer = initializer_constant(),
                                             trainable = TRUE
                                           )
                                           
                                         }
                                         
                                       },
                                       
                                       call = function(x, mask = NULL) {
                                         
                                         if (self$dropout > 0)
                                           x <- k_dropout(x, level = self$dropout)
                                         
                                         x <- k_dot(x, self$weights)
                                         
                                         y <- k_batch_dot(self$adj, x, axes = list(2L, 1L))
                                         
                                         if (self$use_bias) {
                                           
                                           y <- k_bias_add(y, self$bias)
                                           
                                         }
                                         
                                         outputs <- self$activation(y)
                                         
                                         return(outputs)
                                       },
                                       
                                       compute_output_shape = function(input_shape) {
                                         
                                         output_shape <- list(input_shape[[1]],
                                                              input_shape[[2]], 
                                                              self$num_filters)
                                         
                                       }
                                       
                                     ),
                                     
                                     lock_objects = FALSE
)

#' Graph Convolutional layer
#'
#' @param object        (a keras tensor) The input
#' @param num_filters   (integer) the number of output features for each node
#' @param adj           (matrix-like tensor) The (normalized) adjacency matrix
#' @param dropout       (numeric) The dropout rate. Default = 0.
#' @param activation    (activation function in keras) The activation function for the layer.
#' @param use_bias      (logical) Use bias in this layer? Default = TRUE.
#' @param name          (character) Name of the layer.
#' @param trainable     (logical) Can this layer be trained? Default = TRUE.
#'
#' @return A layer that can be concatenated with other layers in keras.
#' @export
#'
layer_gconv <- function(object, 
                        num_filters, 
                        adj,
                        dropout = 0, 
                        activation = activation_linear,
                        use_bias = TRUE,
                        name = NULL, 
                        trainable = TRUE) {
  
  create_layer(GraphConvolutionLayer, object, list(
    num_filters = as.integer(num_filters),
    adj = adj,
    activation = activation,
    dropout = dropout,
    use_bias = use_bias,
    name = name,
    trainable = trainable
  ))
  
}
