import random
import streamlit as st
import requests
import pandas as pd
import json
import pickle
import regex
from underthesea import word_tokenize, pos_tag, sent_tokenize
from pyvi import ViTokenizer
import re
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import joblib
import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
}

# Define the URL of the API
url = "https://tiki.vn/api/v2/products?q="
search_query = "may tinh bang"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
}

# Load the trained model and vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('random_forest_model.pkl')

def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        # ...
        ###### DEL wrong words
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '
    document = new_sentence
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    #...
    return document

# Normalize unicode Vietnamese
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Unicode Vietnamese
def covert_unicode(txt):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], txt)

def find_words(document, list_of_words):
    document_lower = document.lower()
    word_count = 0
    word_list = []

    for word in list_of_words:
        if word in document_lower:
            word_count += document_lower.count(word)
            word_list.append(word)

    return word_count, word_list

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()

neutral_words = ["chấp nhận được", "trung bình", "bình thường", "tạm ổn", "trung lập", "có thể"
                 "không nổi bật", "đủ ổn", "đủ tốt", "có thể chấp nhận", "bình thường",
                 "thường xuyên", "tương đối", "hợp lý", "tương tự",
                 "có thể sử dụng", "bình yên", "bình tĩnh", "không quá tệ", "trung hạng",
                 "có thể điểm cộng", "dễ chấp nhận", "không phải là vấn đề",
                 "không phản đối", "không quá đáng kể", "không gây bất ngờ", "không tạo ấn tượng", "có thể chấp nhận",
                 "không gây sốc", "tương đối tốt", "không thay đổi", "không quá phức tạp", "không đáng kể",
                 "chấp nhận", "có thể dễ dàng thích nghi", "không quá cầu kỳ", "không cần thiết", "không yêu cầu nhiều", "không gây hại",
                 "không có sự thay đổi đáng kể", "không rõ ràng", "không quá phê bình", "không đáng chú ý", "không đặc biệt",
                 "không quá phức tạp", "không gây phiền hà", "không đáng kể", "không gây kích thích"]

negative_words = [
    "kém", "tệ", "đau", "xấu", "bị","rè", "ồn",
    "buồn", "rối", "thô", "lâu", "sai", "hư", "dơ", "không có"
    "tối", "chán", "ít", "mờ", "mỏng", "vỡ", "hư hỏng",
    "lỏng lẻo", "khó", "cùi", "yếu", "mà", "không thích", "không thú vị", "không ổn",
    "không hợp", "không đáng tin cậy", "không chuyên nghiệp", "nhầm lẫn"
    "không phản hồi", "không an toàn", "không phù hợp", "không thân thiện", "không linh hoạt", "không đáng giá",
    "không ấn tượng", "không tốt", "chậm", "khó khăn", "phức tạp", "bị mở", "bị khui", "không đúng", "không đúng sản phẩm",
    "khó hiểu", "khó chịu", "gây khó dễ", "rườm rà", "khó truy cập", "bị bóc", "sai sản phẩm",
    "thất bại", "tồi tệ", "khó xử", "không thể chấp nhận", "tồi tệ","không rõ ràng", "giảm chất lượng",
    "không chắc chắn", "rối rắm", "không tiện lợi", "không đáng tiền", "chưa đẹp", "không đẹp"
]

positive_words = [
    "thích", "tốt", "xuất sắc","đúng", "tuyệt vời", "tuyệt hảo", "đẹp", "ổn"
    "hài lòng", "ưng ý", "hoàn hảo", "chất lượng", "thú vị", "nhanh"
    "tiện lợi", "dễ sử dụng", "hiệu quả", "ấn tượng",
    "nổi bật", "tận hưởng", "tốn ít thời gian", "thân thiện", "hấp dẫn",
    "gợi cảm", "tươi mới", "lạ mắt", "cao cấp", "độc đáo",
    "hợp khẩu vị", "rất tốt", "rất thích", "tận tâm", "đáng tin cậy", "đẳng cấp",
    "hấp dẫn", "an tâm", "không thể cưỡng_lại", "thỏa mãn", "thúc đẩy",
    "cảm động", "phục vụ tốt", "làm hài lòng", "gây ấn tượng", "nổi trội",
    "sáng tạo", "quý báu", "phù hợp", "tận tâm",
    "hiếm có", "cải thiện", "hoà nhã", "chăm chỉ", "cẩn thận",
    "vui vẻ", "sáng sủa", "hào hứng", "đam mê", "vừa vặn", "đáng tiền"
]

# List of words with negative meanings
negation_words = ["không", "nhưng", "tuy nhiên", "mặc dù", "chẳng", "mà", 'kém', 'giảm']

positive_emojis = [
    "😄", "😃", "😀", "😁", "😆",
    "😅", "🤣", "😂", "🙂", "🙃",
    "😉", "😊", "😇", "🥰", "😍",
    "🤩", "😘", "😗", "😚", "😙",
    "😋", "😛", "😜", "🤪", "😝",
    "🤗", "🤭", "🥳", "😌", "😎",
    "🤓", "🧐", "👍", "🤝", "🙌", "👏", "👋",
    "🤙", "✋", "🖐️", "👌", "🤞",
    "✌️", "🤟", "👈", "👉", "👆",
    "👇", "☝️"
]

# Count emojis positive and negative
negative_emojis = [
    "😞", "😔", "🙁", "☹️", "😕",
    "😢", "😭", "😖", "😣", "😩",
    "😠", "😡", "🤬", "😤", "😰",
    "😨", "😱", "😪", "😓", "🥺",
    "😒", "🙄", "😑", "😬", "😶",
    "🤯", "😳", "🤢", "🤮", "🤕",
    "🥴", "🤔", "😷", "🙅‍♂️", "🙅‍♀️",
    "🙆‍♂️", "🙆‍♀️", "🙇‍♂️", "🙇‍♀️", "🤦‍♂️",
    "🤦‍♀️", "🤷‍♂️", "🤷‍♀️", "🤢", "🤧",
    "🤨", "🤫", "👎", "👊", "✊", "🤛", "🤜",
    "🤚", "🖕"
]

# Preoricessing Function
def preprocess_input(input_text, emoji_dict, teen_dict, wrong_lst, neutral_words, negative_words, positive_words, negation_words, positive_emojis, negative_emojis, stopwords_lst):
    # Step1 1: Apply text process
    processed_text = process_text(input_text, emoji_dict, teen_dict, wrong_lst)

    # Step 2: Convert unicode
    processed_text = covert_unicode(processed_text)

    # Step 3: Add new features (Support for model)
    neutral_word_count = find_words(processed_text, neutral_words)[0]
    negative_word_count = find_words(processed_text, negative_words)[0]
    positive_word_count = max(find_words(processed_text, positive_words)[0] - find_words(processed_text, negation_words)[0],0)
    positive_emoji_count = find_words(processed_text, positive_emojis)[0]
    negative_emoji_count = find_words(processed_text, negative_emojis)[0]

    # Step 4: remove stopwords
    tokenized_text = word_tokenize(processed_text, format="text")
    tokenized_text = remove_stopword(tokenized_text, stopwords_lst)

    # Step 5: Apply POS tagging
    tokenized_text = ViTokenizer.tokenize(tokenized_text)
    tokenized_text = re.sub(r'\.', '', tokenized_text)

    # Add in dictionary
    processed_data = {
        "processed_text": tokenized_text,
        "neutral_word_count": neutral_word_count,
        "negative_word_count": negative_word_count,
        "positive_word_count": positive_word_count,
        "positive_emoji_count": positive_emoji_count,
        "negative_emoji_count": negative_emoji_count
    }

    return processed_data

def predict_sentiment(user_input):
    # Step 1: Preprocess input
    processed_data = preprocess_input(user_input, emoji_dict, teen_dict, wrong_lst, neutral_words, negative_words, positive_words, negation_words, positive_emojis, negative_emojis, stopwords_lst)

    # Step 2: Add new features
    processed_text = processed_data['processed_text']
    feature_values = [
        processed_data['neutral_word_count'],
        processed_data['negative_word_count'],
        processed_data['positive_word_count'],
        processed_data['positive_emoji_count'],
        processed_data['negative_emoji_count']
    ]

    # Step 3: Vector text
    text_vectorized = vectorizer.transform([processed_text])

    # Step 4: Combine Vector vs Features
    features_combined = hstack((text_vectorized, np.array(feature_values).reshape(1, -1)))

    # Step 5: Predict Function
    prediction = model.predict(features_combined)

    return prediction[0]

def display_comments(product_id):
    product_comments = [comment for comment in st.session_state.comments if comment["product_id"] == product_id]
    for comment in product_comments:
        st.write(f"product id: {comment['product_id']}")
        st.write(f"Customer Name: {comment['username']}")
        st.write(f"Rating: {comment['rating']} star")
        st.write(f"Content: {comment['content']}")

        prediction = predict_sentiment(comment['content'])
        if prediction == 'positive':
            st.markdown(f"<h1 style='font-size:20px;'>Predict: 😊</h1>", unsafe_allow_html=True)
        elif prediction == 'neutral':
            st.markdown(f"<h1 style='font-size:20px;'>Predict: 😐</h1>", unsafe_allow_html=True)
        else: # Assuming 'negative' is the only other value returned by the predict function
            st.markdown(f"<h1 style='font-size:20px;'>Predict: 😢</h1>", unsafe_allow_html=True)
        # prediction = random.choice(["happy", "sad"])
        # st.write(f"Predict: {'😊' if prediction == 'happy' else '😢'}") # '😊' if prediction == 'happy' else '😢'


# Add sidebar
st.sidebar.header('Sentiment analysis',divider='rainbow')

sidebar_option = st.sidebar.selectbox(
    "Select Options",
    ("Introduction", "🔍 Find your products", "Input manual")
)

st.sidebar.markdown(f"<h1 style='font-size:17px;'>😊 Positive</h1>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h1 style='font-size:17px;'>😐 Neutral</h1>", unsafe_allow_html=True)
st.sidebar.markdown(f"<h1 style='font-size:17px;'>😢 Negative</h1>", unsafe_allow_html=True)


if sidebar_option == "Introduction":
    st.title("Introduction about application")
    st.header('User guide:', divider='rainbow')
    st.subheader('1. Find your products:', divider='rainbow')
    st.write("   - User input name of your products.")
    st.write("   - Click on the name of product to show all of comments.")
    st.write("   - Model predict the comment.")
    st.write("   - Compare model with content in comment.")
    st.write("   - Compare rating with the prediction.")

    st.subheader('2. Input comment manual:', divider='rainbow')
    st.write("   - User input a comment.")
    st.write("   - The model predict this comment is positive, neutral or negative")

    st.header('My contact:', divider='rainbow')
    st.write("Trong, NguyenThanh")

    st.header('Source:', divider='rainbow')
    st.write("eCommece: Tiki and Sendo.")
    st.write("Data Scrapping from Tiki.")

    st.header('Model', divider='rainbow')
    st.write("Random Forest")
    st.write("Accuracy: More than 95%")

elif sidebar_option == "🔍 Find your products":
    st.title("Find your products on Tiki ECommerce")

    search_query = st.text_input("Find your products:")
    if st.button("Find 🔍"):
        st.write(f"Results: {search_query}")

    url = "https://tiki.vn/api/v2/products?q=" + search_query

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes

        data = response.json()    

    except requests.exceptions.RequestException as e:
        st.write(f"An error occurred: {e}")
        data = {"data": []}
    except json.JSONDecodeError as e:
        st.write(f"Error decoding JSON: {e}")
        data = {"data": []}
    
    # Extracting the necessary information and creating a DataFrame
    review_link = 'https://tiki.vn/api/v2/reviews?product_id='
    # Extracting the necessary information and creating a list of dictionaries
    product_list = []
    for item in data["data"]:
        product = {
            "product_id": str(item["id"]),
            "name": item["name"],
            "sold": item.get("quantity_sold", {}).get("value", "N/A"),
            "price": item.get("price", "N/A"),
            "image": item.get("thumbnail_url", "N/A"),
            "review_count": item.get("review_count", "N/A"),
            "rating_average": item.get("rating_average", "N/A"),
            "link_comment": review_link + str(item["id"])
        }
        product_list.append(product)
    products = product_list

    current_product_id = None

    # Check 'comments' exits in st.session_state yet?
    if 'comments' not in st.session_state:
        st.session_state.comments = []

    # Get comments data based on products and column link_comment
    comments = []
    for product in products:
        review_url = product['link_comment']
        try:
            review_response = requests.get(review_url, headers=headers)
            review_response.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes

            review_data = review_response.json()
            
            comments_productid = [
                {
                    "product_id": product["product_id"],
                    "username": review.get("created_by", {}).get("name", "Anonymous"),
                    "rating": review.get("rating", 0),
                    "content": review.get("content", "No content"),
                }
                for review in review_data.get("data", [])
            ]
            for comment in comments_productid:
                if comment not in st.session_state.comments:
                    st.session_state.comments.append(comment)
        except requests.exceptions.RequestException as e:
            st.write(f"An error occurred: {e}")
        except json.JSONDecodeError as e:
            st.write(f"Error decoding JSON: {e}")

    # Display all the comments 
    # for comment in comments:
    #    st.write(f"Product ID: {st.session_state.comments['product_id']}")
    #    st.write(f"Username: {st.session_state.comments['username']}")
    #    st.write(f"Rating: {st.session_state.comments['rating']} stars")
    #    st.write(f"Content: {st.session_state.comments['content']}")

    # Debug: print the number of comments fetched
    st.write(f"Number of comments fetched: {len(st.session_state.comments)}")

    if 'current_product_id' not in st.session_state:
        st.session_state.current_product_id = None

    cols = st.columns(2)
    for i, product in enumerate(products):
        with cols[i % 2]:
            st.image(product["image"])
            if st.button(product["name"], key=f"product_{product['product_id']}"):
                st.session_state.current_product_id = product["product_id"]
                display_comments(product["product_id"])
            # Điều chỉnh cỡ chữ và định dạng chữ cho các trường 'sold' và 'price'
            st.markdown(f"<p style='font-size: small;'>Sold: {product['sold']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: small;'>Reviews: {product['review_count']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: small;'>Rate: {product['rating_average']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: small;'>Product id: {product['product_id']}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size: larger; font-weight: bold;'>Price: {product['price']}</p>", unsafe_allow_html=True)
            
            with st.expander("Add new comments"):
                with st.form(key=f'add_comment_{product["product_id"]}'):
                    st.text_input('Name', key=f'username_{product["product_id"]}')
                    st.slider('Rating', min_value=0, max_value=5, key=f'rating_{product["product_id"]}')
                    st.text_area('Content', key=f'content_{product["product_id"]}')
                    
                    if st.form_submit_button('Add comment'):
                        new_comment = {
                            "product_id": product['product_id'],
                            "username": st.session_state[f'username_{product["product_id"]}'],
                            "rating": st.session_state[f'rating_{product["product_id"]}'],
                            "content": st.session_state[f'content_{product["product_id"]}']
                        }
                        st.session_state.comments.append(new_comment)
                        st.write('Your comment has been added successfully')
                        display_comments(st.session_state.current_product_id)

elif sidebar_option == "Input manual":
    st.title("Predict comment from User")
    user_input = st.text_area("Type your comment here")
    if st.markdown("""
        <style>
        .custom-button {
            font-size: 20px;
            height: 50px;
            width: 200px;
            border: 2px solid white;
            border-radius: 20px;
            text-align: center;
            line-height: 50px; /* This will vertically center the text */
            padding-left: 1px; /* This will horizontally align the text to the left */
        }
        </style>
        <button class="custom-button" onclick="handleClick()">🔍 Predict</button>
        <script>
            function handleClick() {
                // Logic to handle button click
            }
        </script>
        """, unsafe_allow_html=True):
        prediction = predict_sentiment(user_input)
        if prediction == 'positive':
            st.markdown(f"<h1 style='font-size:30px;'>Predict: 😊</h1>", unsafe_allow_html=True)
        elif prediction == 'neutral':
            st.markdown(f"<h1 style='font-size:30px;'>Predict: 😐</h1>", unsafe_allow_html=True)
        else: # Assuming 'negative' is the only other value returned by the predict function
            st.markdown(f"<h1 style='font-size:30px;'>Predict: 😢</h1>", unsafe_allow_html=True)