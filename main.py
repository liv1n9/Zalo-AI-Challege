from feature_extraction import extract_feature
from regression_model import Hotness
from regression_model import create_feature

# Extract feature from audio
extract_feature('train')
extract_feature('test')

# Calculate hotness of artists and composers
hotness = Hotness()
hotness.compute()

# Create feature vector
create_feature('train')
create_feature('test')
