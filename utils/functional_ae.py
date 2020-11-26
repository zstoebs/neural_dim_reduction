### autoencoder builder function

def create_ae(init='uniform',activation='relu',optimizer='adam',ret_comp=False):
    """
    sizes: list of layer sizes in decreasing order until encoding --> mirrored for decoding
    ret_all: returns components of the autoencoder as well if True
    """
    global SIZES
    
    if len(SIZES) < 2:
        raise ValueError("Autoencoder must have at least 3 layers.")
    
    enc_inp = Input(shape=(SIZES[0],))
    enc = enc_inp
    for size in SIZES[1:]:
        enc = Dense(size,kernel_initializer=init,activation=activation)(enc)
        enc = Dropout(rate=0.2)(enc)
        
    dec_inp = Input(shape=(SIZES[-1],))
    dec = dec_inp
    for size in reversed(SIZES[1:-1]):
        dec = Dropout(rate=0.2)(dec)
        dec = Dense(size,kernel_initializer=init,activation=activation)(dec)
    
    dec = Dropout(rate=0.2)(dec)
    dec = Dense(SIZES[0],kernel_initializer=init,activation='sigmoid')(dec)

    encoder = Model(enc_inp,enc,name='encoder')
    decoder = Model(dec_inp,dec,name='decoder')
    autoencoder = Model(enc_inp,decoder(encoder(enc_inp)),name='autoencoder')
    
    autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return [encoder, decoder] if ret_comp else autoencoder
