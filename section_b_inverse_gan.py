from main import model_loader
from main import Encoder

generator, discriminator = model_loader()

inverser = Encoder(resolution=(28, 28), channels=1, compression_size=generator.latent_size)

