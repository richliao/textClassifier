# textClassifier

textClassifierHATT.py has the implementation of [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf). Please see the [my blog](https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/) for full detail. Also see [Keras Google group discussion](https://groups.google.com/forum/#!topic/keras-users/IWK9opMFavQ)

textClassifierConv has implemented [Convolutional Neural Networks for Sentence Classification - Yoo Kim](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf). Please see the [my blog](https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/) for full detail.

textClassifierRNN has implemented bidirectional LSTM and one level attentional RNN. Please see the [my blog](https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/) for full detail.

## update on 6/22/2017 ##
To derive the attention weight which can be useful to identify important words for the classification. Please see my latest update on the post. All you need to do is run a forward pass right before attention layer output. The result is not very promising. I will update the post once I have further result.

---
This repo is forked from [https://github.com/richliao/textClassifier](https://github.com/richliao/textClassifier) and we find some issue [here](https://github.com/richliao/textClassifier/issues/28). So we update the textClassifierHATT with `python 2.7` and `keras 2.0.8`

```
# clone the repo
git clone {repo address}

# install Dependent library
cd textClassifier
pip install -r req.xt

# download imdb train from Kaggle
wget https://www.kaggle.com/c/word2vec-nlp-tutorial/download/labeledTrainData.tsv
# download glove word vector
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip

# train the model
python textClassifierHATT.py 
```



Enjoy！