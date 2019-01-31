from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None 

def lemmatize_tagged_token(token):
    lemmatizer = WordNetLemmatizer()
    word = token[0]
    tag = token[1]
    wntag = get_wordnet_pos(tag)
    if wntag is None:# not supply tag in case of None
        return lemmatizer.lemmatize(word) 
    else:
        return lemmatizer.lemmatize(word, pos=wntag) 
    
