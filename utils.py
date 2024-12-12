import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences


# return a dictionary with image names as keys and cleaned captions as values
def get_clean_captions(file_path):
    caption_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            image_name, caption = line.split(',', 1)

            # 소문자로 변환
            caption = caption.lower().strip()
            # 특수기호 제거
            caption = re.sub(r'[^\w\s]', '', caption)
                # ^: 제외
                # \w: 알파벳, 숫자, 문자
                # \s: 공백, 줄바꿈
            # Removing numbers
            caption = re.sub(r'\d+', '', caption)
            # Removing extra whitespace
            caption = re.sub(r'\s+', ' ', caption).strip()

            #  Adding 'startsymbol' word and 'endsymbol' word to the captions
            caption = 'startsymbol ' + caption + ' endsymbol'

            # caption 저장
            if image_name not in caption_dict:
                caption_dict[image_name] = []
            caption_dict[image_name].append(caption)

    # 파일에서 첫 행에 들어있는 'image' 키 제거
    if 'image' in caption_dict:
        del caption_dict['image']
    
    return caption_dict

# functions to create datasets
def training_data(dict_keys, caption_dict, features, tokenizer, max_length, vocab_size, batch_size):
    image_features, income_seqs, outcome_seqs = list(), list(), list()
    n = 0
    while 1:
      for dict_key in dict_keys:
          captions = caption_dict[dict_key]
          for caption in captions:
            # 캡션 토크나이징 및 시퀀스 처리
            seq = tokenizer.texts_to_sequences([caption])[0]
            for j in range(1, len(seq)):
                in_seq, out_seq = seq[:j], seq[j]
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]


                image_features.append(features[dict_key])
                income_seqs.append(in_seq)
                outcome_seqs.append(out_seq)

          n += 1 # counter for the number of elements in a batch
          if n == batch_size:
                yield (np.array(image_features), np.array(income_seqs)), np.array(outcome_seqs) # 튜플로 수정
                image_features, income_seqs, outcome_seqs = list(), list(), list()
                n = 0

def create_dataset(dict_keys, caption_dict, features, tokenizer, max_length, vocab_size, batch_size):
    output_signature = (
        (
            tf.TensorSpec(shape=(None, 1280), dtype=tf.float32),  # image_features
            tf.TensorSpec(shape=(None, max_length), dtype=tf.int32),  # income_seqs
        ),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32),  # outcome_seqs
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: training_data(dict_keys, caption_dict, features, tokenizer, max_length, vocab_size, batch_size),
        output_signature=output_signature
    )
    return dataset.prefetch(tf.data.AUTOTUNE) # 성능 향상을 위한 prefetch 추가

# functions to generate captions by using the trained model
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startsymbol'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length, padding='post')[0]
        image = tf.reshape(image, (1, 1280))
        sequence = tf.expand_dims(sequence, axis=0)
        yhat = model([image, sequence], training=False).numpy() # training=False 추가
        yhat = np.argmax(yhat, axis=-1) # axis=-1 추가
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endsymbol':
            break
    return in_text

def clean_predicted_text(text):
    tokens = text.split()
    # 시작 및 끝 심볼 제거
    if tokens[0] == 'startsymbol':
        tokens = tokens[1:]
    if tokens[-1] == 'endsymbol':
        tokens = tokens[:-1]
    
    # 입력 타입에 따라 반환
    return " ".join(tokens)

def visualize_caption(image_name, tokenizer, features, caption_dict, max_length, model1, model2, model3, output_name="output"):
    """
    이미지와 캡션을 함께 시각화합니다.

    Args:
        image_path (str): 이미지 파일 경로
        original_caption (str): 원본 캡션
        generated_caption (str): 생성된 캡션
    """
    try:
        img_path = os.path.join('./dataset', "Images", image_name)
        img = Image.open(img_path)
        # matplotlib을 사용하여 이미지와 캡션을 함께 표시
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1x2 그리드, figsize 조정

        # 왼쪽: 이미지 표시
        ax1.imshow(img)
        ax1.axis('off')  # 축 숨기기
        ax1.set_title("Image")

        # 오른쪽: 캡션 표시
        captions = caption_dict[image_name]

        ax2.axis('off')  # 축 숨기기
        # original captions
        ax2.text(0.0, 0.9, "Original Captions:", fontsize=10, fontweight='bold')
        idx=0
        for caption in captions:
            caption = clean_predicted_text(caption)
            ax2.text(0.0, 0.85-idx*0.05, caption, fontsize=10, wrap=True) # wrap=True 추가
            idx+=1

        # predict the caption
        y_pred1 = predict_caption(model1, features[image_name], tokenizer, max_length)
        y_pred2 = predict_caption(model2, features[image_name], tokenizer, max_length)
        y_pred3 = predict_caption(model3, features[image_name], tokenizer, max_length)
        y_pred1 = clean_predicted_text(y_pred1)
        y_pred2 = clean_predicted_text(y_pred2)
        y_pred3 = clean_predicted_text(y_pred3)

        ax2.text(0.0, 0.6, "Generated by model_1:", fontsize=10, fontweight='bold')
        ax2.text(0.0, 0.55, y_pred1, fontsize=10, wrap=True)
        ax2.text(0.0, 0.5, "Generated by model_2:", fontsize=10, fontweight='bold') 
        ax2.text(0.0, 0.45, y_pred2, fontsize=10, wrap=True)
        ax2.text(0.0, 0.4, "Generated by model_3:", fontsize=10, fontweight='bold') 
        ax2.text(0.0, 0.35, y_pred3, fontsize=10, wrap=True)
        
        output_path = output_name + ".png"
        plt.tight_layout()
        plt.savefig(output_path)  # 파일로 저장
        plt.close(fig) # 메모리 누수 방지

    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
