import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 텐소플로우 로그메시지 출력수준
import warnings
warnings.filterwarnings('ignore')


import pickle
import random

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam

from utils import *
from build_model import lstm_model
from image_features import extract_features

# hyperparameter
MAIN_PATH = "./dataset"
CAPTIONS_PATH = "./dataset/captions.txt"

NUM_TEST_IMAGES = -50

EPOCHS = 30
BATCH_SIZE = 64


# extract image features using EfficientNetV2S model
# extract_features(MAIN_PATH, 'features.pkl')

# open the saved pickle file
with open('features.pkl', 'rb') as f:
    features = pickle.load(f)
print('='*60)
print('STEP 1: Extracted the image features.')
# create the dictionary of captions with image names as keys
caption_dict = get_clean_captions(CAPTIONS_PATH)

# setup tokenizer
tokenizer = Tokenizer()
all_captions = []
for key in caption_dict:
    for caption in caption_dict[key]:
        all_captions.append(caption)

tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
MAX_LENGTH = max(len(caption.split()) for caption in all_captions) + 1
print('='*60)
print('STEP 2: Tokenizing')
print("이미지 캡션의 최대 길이: ", MAX_LENGTH)
print("캡션 데이터 속의 단어 개수: ", vocab_size)

# split data into train and test
image_names = list(caption_dict.keys())
image_names_shuffled = random.sample(image_names, len(image_names))
train_keys = image_names_shuffled[:NUM_TEST_IMAGES]
test_keys = image_names_shuffled[NUM_TEST_IMAGES:]
print('='*60)
print('STEP 3: Modeling')
# build model
model = lstm_model(MAX_LENGTH, vocab_size)
print(model.summary())

# compile
model.compile(
    optimizer=Adam(learning_rate=0.0006),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print('='*60)
print('STEP 4: Training')

steps = 5*len(train_keys) // BATCH_SIZE
# training
for i in range(EPOCHS):
    # create data generator using tf.data.Dataset
    train_dataset = create_dataset(train_keys, caption_dict, features, tokenizer, MAX_LENGTH, vocab_size, BATCH_SIZE)
    history = model.fit(train_dataset, epochs=1, steps_per_epoch=steps, verbose=0)
    print(f'EPOCH {i}, Loss: ', history.history['loss'])
    # save the model
    if i+1 == 10:    
        model.save('saved_model_10.keras')
    elif i+1 == 20:    
        model.save('saved_model_20.keras')

model.save('saved_model_final.keras')

print('='*60)
print('Finished Training!')

print('='*60)
print('STEP 5: Test Images')

model1 = load_model('saved_model_10.keras')
model2 = load_model('saved_model_20.keras')
model3 = load_model('saved_model_final.keras')

for jdx in range(10):
    output_name = "./test_images/output" + str(jdx)
    idx = random.randint(0, len(test_keys))
    visualize_caption(test_keys[idx], tokenizer, features, caption_dict, MAX_LENGTH, model1, model2, model3, output_name)

print('='*60)
print('DONE!')