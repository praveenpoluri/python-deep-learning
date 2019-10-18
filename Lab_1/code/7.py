# 1) Reading data from file
d= open('input.txt','r')
input=d.read()

import nltk
nltk.download('punkt')

# 2) text tokenization into words and lemmatization applied on each word.

# words tokenization
word_Tkns = nltk.word_tokenize(input)
print("2) text tokenization into words and lemmatization applied on each word.\n")
for w in word_Tkns:
    print(w)
print("\n")

# Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lmt = WordNetLemmatizer()
for word_Tkns in word_Tkns:
    print(lmt.lemmatize(word_Tkns))
print("\n")


# 3) Trigrams
tr_dic = {}
trgrams = nltk.trigrams(input.split())
print("3) Trigrams\n")
for item in trgrams:
    dicKey=' '.join(item)
    # print(dicKey)
    if dicKey in tr_dic.keys():
        tr_dic.update({dicKey:tr_dic[dicKey]+1})
    else:
        tr_dic[dicKey]=1
print(tr_dic,"\n")

#4) to get top 10 values from dictionary

from heapq import nlargest

topten = nlargest(10, tr_dic, key=tr_dic.get)
print("4) top 10 values from dictionary:")
for val in topten:
    print(val, ":", tr_dic.get(val))
print("\n")

# 5,6) sentence tokenization and Finding all the sentences with the mostly repeated tri-grams


dsfs= "we need to"
tokens_sent = nltk.sent_tokenize(input)
max_val = max(tr_dic, key=tr_dic.get)
combined_string=''
print("Tasks 5,6 Combined: ")
print("All sentences with mostly repeated tri-grams: \n")

# 7,8) sentences extraction, concatenation and printing the result

for s in tokens_sent:
    if max_val in s:
       print(s)
       combined_string = combined_string+s
print("\n")
print("Task 7,8 Combined")
print("Concatenated Result :\n")
print(combined_string)