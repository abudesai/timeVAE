# TimeVAE for Synthetic Timeseries Data Generation
TimeVAE implementation in keras/tensorflow implementation of timevae: 

TimeVAE is used for synthetic time-series data generation. See paper:

https://arxiv.org/abs/2111.08095

The methodology uses the Variational Autoencoder architecture. The decoder architecture is modified to include interpretable components of time-series, namely, level, trend, and seasonality. 

'vae_conv_I_model.py' script contains the interpretable version of TimeVAE. See class 'VariationalAutoencoderConvInterpretable'. 

'vae_conv_model.py' contains the base version of TimeVAE. See class 'VariationalAutoencoderConv'

The VariationalAutoencoderConvInterpretable in 'vae_conv_I_model.py' can also be used as base version by disabling the interpretability-related arguments during class initialization. 

See script test_vae for usage of the TimeVAE model. 

Note that 'vae_base' script contains an abstract super-class.  It doesnt actually represent TimeVAE-Base. 
