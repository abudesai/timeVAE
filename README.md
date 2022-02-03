# timeVAE
TimeVAE implementation in keras/tensorflow

implementation of timevae: 
https://arxiv.org/abs/2111.08095

vae_conv_I_model script contains the interpretable version of TimeVAE. See class 'VariationalAutoencoderConvInterpretable'. 

vae_conv_model contains the base version of TimeVAE. See class 'VariationalAutoencoderConv'

The VariationalAutoencoderConvInterpretable can also be used as base version by disabling the interpretability-related arguments during class initialization. 

See script test_vae for usage of the TimeVAE model. 

Note that 'vae_base' script contains an abstract super-class.  It doesnt actually represent TimeVAE-Base. 
