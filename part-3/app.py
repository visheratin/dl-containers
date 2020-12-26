import pickle
from typing import List
import os
import urllib

import streamlit as st
from transformers import AutoTokenizer
import numpy as np
from scipy.spatial import distance
import onnxruntime

class DataItem(object):
    def __init__(self, title: str, abstract: str, hash: np.ndarray) -> None:
        self.title = title
        self.abstract = abstract
        self.hash = hash

DATA_ITEMS = {
    "model": {
        "fname": "news.onnx",
        "url": "https://www.dropbox.com/s/gw6agt1227hjx0v/news.onnx?dl=1"
    },
    "data": {
        "fname": "news.pkl",
        "url": "https://www.dropbox.com/s/dmniqs5vdbqr06c/news.pkl?dl=1"
    }
}

DATA_DIR = "."

def main():
    os.makedirs(DATA_DIR, 0o666, exist_ok=True)
    for k in DATA_ITEMS:
        fpath = os.path.join(DATA_DIR, DATA_ITEMS[k]["fname"])
        download_file(DATA_ITEMS[k]["url"], fpath)
    st.title('Similar news searcher')
    title = st.text_area('Insert title of the target news')
    abstract = st.text_area('Insert abstract of the target news')
    submit = st.button('Get similar news')
    if submit:
        items = process(title, abstract, DATA_DIR)
        for item in items:
            st.subheader(item.title)
            st.write(item.abstract)


@st.cache
def process(title: str, abstract: str, data_dir: str) -> List[DataItem]:
    model = load_model(data_dir)
    tkn = load_tkn()
    data = load_data(data_dir)
    d = extract_tokens(title, abstract, tkn)
    h = hash(d, model)
    return get_similar(h, data)


@st.cache(allow_output_mutation=True)
def load_model(data_dir: str):
    model_path = os.path.join(data_dir, DATA_ITEMS["model"]["fname"])
    model = onnxruntime.InferenceSession(model_path)
    return model


@st.cache(allow_output_mutation=True)
def load_tkn() -> AutoTokenizer:
    return AutoTokenizer.from_pretrained("distilbert-base-uncased")


@st.cache
def load_data(data_dir: str) -> List[DataItem]:
    data = None
    data_path = os.path.join(data_dir, DATA_ITEMS["data"]["fname"])
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_similar(hash: np.ndarray, items: List[DataItem]) -> List[DataItem]:
    return sorted(items, key=lambda x: dist(hash, x.hash))[:10]


def hash(d: np.ndarray, model) -> np.ndarray:
    d = np.expand_dims(d, axis=0)
    input = {model.get_inputs()[0].name: d}
    res = model.run(None, input)
    return res[0][0]


def dist(hash_1: np.ndarray, hash_2: np.ndarray) -> float:
    return distance.cosine(hash_1, hash_2)


def extract_tokens(title: str, abstract: str, tkn: AutoTokenizer) -> np.ndarray:
    maxlen = 100
    title_tokens = tkn.encode_plus(
        title,
        add_special_tokens=True,
        truncation=True,
        max_length=maxlen,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='np',
    )
    abstract_tokens = tkn.encode_plus(
        abstract,
        add_special_tokens=True,
        truncation=True,
        max_length=maxlen,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='np',
    )
    t = np.concatenate((title_tokens['input_ids'],
                   title_tokens['attention_mask']), axis=0)
    a = np.concatenate((abstract_tokens['input_ids'],
                   abstract_tokens['attention_mask']), axis=0)
    r = np.concatenate((t, a))
    return r


def download_file(file_url, file_path):
    if os.path.exists(file_path):
        return
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s ..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(file_url) as response:
                info = response.info()
                length = int(info["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))
    except Exception as e:
        st.error(e)
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


if __name__ == "__main__":
    main()