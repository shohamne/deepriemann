#set -x

#python tf_conv_mnist.py full
#python tf_conv_mnist.py 1024
#python tf_conv_mnist.py 512
#python tf_conv_mnist.py 256
#python tf_conv_mnist.py 128
#python tf_conv_mnist.py 64
#python tf_conv_mnist.py 64 riemannian
#python tf_conv_mnist.py 32 parameterization
#python tf_conv_mnist.py 32 riemannian

python tf_conv_mnist.py full
python tf_conv_mnist.py 16 parameterization 0.4
python tf_conv_mnist.py 16 riemannian 0.4
python tf_conv_mnist.py 8 parameterization 0.4
python tf_conv_mnist.py 8 riemannian 0.4
python tf_conv_mnist.py 4 parameterization 0.4
python tf_conv_mnist.py 4 riemannian 0.4
python tf_conv_mnist.py 2 parameterization 0.4
python tf_conv_mnist.py 2 riemannian 0.4

#python tf_conv_mnist.py 4 parameterization 0.2
#python tf_conv_mnist.py 4 riemannian 0.2
#python tf_conv_mnist.py 4 parameterization 0.4
#python tf_conv_mnist.py 4 riemannian 0.4
#python tf_conv_mnist.py 4 parameterization 0.6
#python tf_conv_mnist.py 4 riemannian 0.6
#python tf_conv_mnist.py 4 parameterization 0.8
#python tf_conv_mnist.py 4 riemannian 0.8
#python tf_conv_mnist.py 4 parameterization 0.9
#python tf_conv_mnist.py 4 riemannian 0.9
#python tf_conv_mnist.py 4 parameterization 0.95
#python tf_conv_mnist.py 4 riemannian 0.95
#python tf_conv_mnist.py 4 parameterization 0.99
#python tf_conv_mnist.py 4 riemannian 0.99

#python tf_conv_mnist.py 2 parameterization 0.2
#python tf_conv_mnist.py 2 riemannian 0.2
#python tf_conv_mnist.py 2 parameterization 0.4
#python tf_conv_mnist.py 2 riemannian 0.4
#python tf_conv_mnist.py 2 parameterization 0.6
#python tf_conv_mnist.py 2 riemannian 0.6
#python tf_conv_mnist.py 2 parameterization 0.8
#python tf_conv_mnist.py 2 riemannian 0.8
#python tf_conv_mnist.py 2 parameterization 0.9
#python tf_conv_mnist.py 2 riemannian 0.9



#python tf_conv_mnist.py 1 parameterization
#python tf_conv_mnist.py 1 riemannian
