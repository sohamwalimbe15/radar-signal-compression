i want to run these following experiments and i will give you the gist of it here:
consider original image size is (256,256,2) and the size for output will be the same.
consider 2 latest spaces, one with some spatial dim for ex. (7,7,1024) and second one smaller (1,1,2048).
let's call latent_dim1 and latent_dim2.
our autoencoders will be divided into 4 parts:
1. E1 (from image space to latent_dim1)
2. E2 (from latent_dim1 to latent_dim2)
3. D1 (from latent_dim2 to latent_dim1)
4. D2 (from latent_dim1 to image space)

experiment 1:
the complete autoencoder E1E2D1D2 using baseline architecture trained completely for reconstruction.

experiment 2:
the complete autoencoder E1E2D1D2 using complex architecture trained completely for reconstruction.

experiment 3:
here is the difficult part, the procedure will look like this, this is with the baseline architecture.
step 1: i will take the trained E1 and E2 from the exp 1, and train a conditional flow matching that uses the condition to be the vector in latent_dim2 which is the output of E1E2 and samples noise from latent_dim1 and generates the output of E1. E1 and E2 are frozen here, just flow matching FM model is being trained.
step 2: i will train another autoencoder but this one is just E1D2 (so till the latent_dim1 space) while E1 is frozen, so only D2 gets trained and these weights will be used to initialize D2 in the next step.
step 3: i will now train the autoencoder to be E1E2 then FM then D2 while E1, E2 and FM are frozen while D2 is being trained with its init coming from step 2. and this will be my final model for experiment 3.

experiment 4:
it is the same as experiment 3 but with complex architecture.