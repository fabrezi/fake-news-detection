import pandas as pd
import re
import string

####################################################################
#remaining parts:
#1. implement stop words, puntctuation
#2. bayesian theorem math model/ classifier
#3. train? fuwk
#4. raw output
# number of words ~ 4000000
####################################################################

stop_words = [
"a", "about", "above", "across", "after", "afterwards",
"again", "all", "almost", "alone", "along", "already", "also",
"although", "always", "am", "among", "amongst", "amoungst", "amount", "an",
"and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "as", "at", "be", "became",
"because", "become","becomes", "becoming", "been", "before", "behind", "being", "beside", "besides", "between",
"beyond", "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe", "do", "done", "each",
"eg", "either", "else", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "find","for",
"found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein",
"hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in", "indeed", "is", "it", "its", "itself", "keep", "least",
"less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "name",
"namely", "neither", "never", "nevertheless", "next","no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often",
"on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part","perhaps", "please",
"put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so", "some", "somehow", "someone",
"something", "sometime", "sometimes", "somewhere", "still", "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence", "there"
"thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
"this", "those", "though", "through", "throughout",
"thru", "thus", "to", "together", "too", "toward", "towards",
"under", "until", "up", "upon", "us",
"very", "was", "we", "well", "were", "what", "whatever", "when",
"whence", "whenever", "where", "whereafter", "whereas", "whereby",
"wherein", "whereupon", "wherever", "whether", "which", "while",
"who", "whoever", "whom", "whose", "why", "will", "with",
"within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves"
]

exclude = list(string.punctuation) + stop_words + []
remove = re.compile('[%s]' % string.punctuation)

df = pd.read_excel('C:\\Users\\farid-PC\\Desktop\\class\\CS6045\\acm_project\\train_fake_news.xlsx')
pd.set_option('display.max_colwidth', 1000)#untruncate the unseen text

f = []
for i, s in enumerate(df['Text']):
    s = s.lower()
    no_nums = re.sub(r'[0-9]+' , ' ', s)
    o = remove.sub('', no_nums)
    line = o.split()
    common = list(set(line).intersection(exclude))
    line = ' '.join(word for word in line if word in common)
    f.append(line)

ndf = pd.DataFrame({'Text': f})
frequency = df.Text.str.split(expand=True).stack().value_counts()# counter
T = 4000000 #microsoft word count ~
word_freq = frequency/T #frequency of the word occurrence in the document

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
 print("word                     P(w)")
 print(word_freq)




#with open('file_name', 'r') as f:
 #for line in f:
  # for word in line.split()
   #   print(word)

