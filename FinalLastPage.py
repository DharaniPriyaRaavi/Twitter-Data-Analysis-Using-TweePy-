from csv import excel

import pandas as pd
import numpy as np
import operator
from collections import OrderedDict
import scipy.stats
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from numpy import NaN
from sklearn.feature_extraction.text import  TfidfVectorizer
import re
from nltk.corpus import words
from collections import OrderedDict
import math
import numpy
# nltk.download('words')
final_output=[]
clusters_data = pd.read_csv("Kmeans CLusters.csv")
tot_tweets=len(clusters_data)
print("tot_tweets", tot_tweets)
# clusters_data.to_csv("Kmeans CLusters.csv")
# clusters_data= pd.read_csv("Kmeans CLusters.csv")

#Step 9 done
df=clusters_data.groupby(['clusters','UserName'])['No.OfRetweets','TweetCount'].sum().reset_index()
df['local_influential_score']=df['No.OfRetweets']/df['TweetCount']
df.to_csv("local_inf_score.csv",index=False)


#step10 for each cluster and number of tweets in the cluster and number of users

df_10=clusters_data.groupby(['clusters']).agg({'UserName': pd.Series.nunique,'TweetCount':'sum'})
print(df_10)


#step 11 14*14 matrix number of common users
df=df.astype(str).groupby(['clusters'],sort=False).agg({'UserName':lambda x: ','.join(x.unique())})


ALLUserList=[]
for i in range(0,len(df["UserName"])):
    list_name = str(df["UserName"][i]).split(",")
    ALLUserList.append(list_name)
#
# for( int i=0;i<n;i++)
#     for int jj=0;j<n;j++

w, h = 14, 14;
commonUsersMatrix = [[0 for x in range(w)] for y in range(h)]
for i in range(0,len(ALLUserList)):
    for j in range(0,len(ALLUserList)):
        count=0
        for m_list in ALLUserList[i]:
            for sub_list in ALLUserList[j]:
                if m_list==sub_list:
                    count=count+1

        commonUsersMatrix[i][j]=count

commonUsersMatrix = np.array(commonUsersMatrix)
np.fill_diagonal(commonUsersMatrix,0)
commonUsersMatrix=np.triu(commonUsersMatrix, k=0)
print("===========CommonUserMatrix==============")
print(commonUsersMatrix)
# print(len(commonUsersMatrix))
# print(commonUsersMatrix[0][12])

#step 12: Jacard Matrix formation
w, h = 14, 14;
JaccardMatrix = [[0 for x in range(w)] for y in range(h)]
for i in range(0,len(commonUsersMatrix)):
    for j in range(0,len(commonUsersMatrix)):
        if commonUsersMatrix[i][j]!=0:
            JaccardMatrix[i][j]=commonUsersMatrix[i][j]/(df_10["UserName"][i+1]+df_10["UserName"][j+1]+commonUsersMatrix[i][j])
        else:
            JaccardMatrix[i][j]=commonUsersMatrix[i][j]


JaccardMatrix= np.array(JaccardMatrix)
print("=======Jaccard Matrix==========")
print(JaccardMatrix)
#Step13
Jaccard_index_per=JaccardMatrix*100
print("=======Jaccard Index Percentage==========")
print(Jaccard_index_per)

#Step 14  Cluster Pair>Threshold value greater than Jaccard percentage   I took threshold as 5.0


df_14 = pd.DataFrame(columns=['cluster pair', 'no of Common users','users', 'Jacard_Index_percentage'])
rows=[];
# print("cluster pair\t no of Cusers \t users\t Jacard_Index_percentage\t")
Jaccard_Threshold=0.25;
for i in range(0,len(Jaccard_index_per)):
    for j in range(0,len(Jaccard_index_per)):
        if Jaccard_index_per[i][j]>Jaccard_Threshold:
           # print(i+1,",",j+1,"\t\t\t",commonUsersMatrix[i][j],"\t\t\t",i+1,"=",df_10["UserName"][i+1],j+1,"=",df_10["UserName"][j+1],"\t\t\t",Jaccard_index_per[i][j])
            row=[str(i + 1) + "," + str(j + 1),commonUsersMatrix[i][j],str(i + 1) + "=" + str(df_10["UserName"][i + 1]) + "," + str(j + 1) + "=" + str(df_10["UserName"][j + 1]),JaccardMatrix[i][j]]
            rows.append(row);

df_14 = pd.DataFrame(rows, columns=['cluster pair', 'no of Common users','users', 'Jacard_Index'])
df_14.to_csv("step14_Threshold.csv",sep=',', encoding='utf-8')

print(df_14)

#To find spearman correlation matrix
df_LIS= pd.read_csv("local_inf_score.csv")

df_LIS=df_LIS.astype(str).groupby(['clusters'],sort=False).agg({'local_influential_score':lambda x: ','.join(x)})
#print(df_LIS)


#print(ALLUserList)


AllInfluenceScores=[];
for i in range(0,len(df_LIS["local_influential_score"])):
    list_score = str(df_LIS["local_influential_score"][i]).split(",")
    AllInfluenceScores.append(list_score)

#print(AllInfluenceScores)

clusterPairs=df_14["cluster pair"].to_list()
# print(clusterPairs)
def spearmans_rank_correlation(xs, ys):
    # Calculate the rank of x's
    xranks = pd.Series(xs).rank(ascending=False)
    yranks = pd.Series(ys).rank(ascending=False)
    sigma=0;
    for i in range(0,len(xs)):
        sigma=sigma+abs(xranks[i]-yranks[i])**2
    sigma=6*sigma
    denominator=(len(xs)**3)-(len(xs))
    spearmanRank=sigma/denominator
    spearmancorrelation=1-spearmanRank
    return spearmancorrelation

spearman_list=[]
filter_clusterpairs=[];
for clusterpair in clusterPairs:
    clusterpair=str(clusterpair).split(",")
    pair1=ALLUserList[int(clusterpair[0])-1]
    pair2=ALLUserList[int(clusterpair[1])-1]
    s1=[]
    s2=[]
    for p1 in range(0,len(pair1)):
        for p2 in range(0,len(pair2)):
            if pair1[p1] == pair2[p2]:
                s1.append(float(AllInfluenceScores[int(clusterpair[0])-1][p1]))
                s2.append(float(AllInfluenceScores[int(clusterpair[1])-1][p2]))
                print(pair1[p1])
    if len(s1)>1:
        print(clusterpair)
        spearman_list.append(spearmans_rank_correlation(s1,s2))
        filter_clusterpairs.append(clusterpair)

f_jaccard=[]
for f_cp in filter_clusterpairs:
    pair1=int(f_cp[0])
    pair2=int(f_cp[1])
    f_jaccard.append(JaccardMatrix[pair1-1][pair2-1])

print(filter_clusterpairs)
print(spearman_list)
print(f_jaccard)
similarityscore=[]
for i in range(0,len(f_jaccard)):
    similarityscore.append(f_jaccard[i]*spearman_list[i])

print(similarityscore)
print("Similarity Scores After Threshold positive values---------->")

Similarity_Threshold=0.00;
df_SimilartyScore = pd.DataFrame(columns=['cluster pair', 'Similarity Score'])
rows=[];
for index in range(0,len(similarityscore)):
    if similarityscore[index]>Similarity_Threshold:  #Threshold similarity score +ve
       row=[filter_clusterpairs[index],similarityscore[index]]
       rows.append(row);

df_SimilartyScore = pd.DataFrame(rows, columns=['cluster pair', 'Similarity Score'])
df_SimilartyScore=df_SimilartyScore.sort_values("Similarity Score",ascending=False).reset_index()
df_SimilartyScore.to_csv("Similarity_Scores.csv",sep=',', encoding='utf-8',index=False)
print(df_SimilartyScore)

#Till Step 20 done ***********************
newCluster=15
merge_nodes_list=[]
merge_nodes_dict={}
updated_cluster=clusters_data
print(tuple(df_SimilartyScore['cluster pair']))

decending_pq={}
for x in range(0,len(df_SimilartyScore['cluster pair'])):
    decending_pq[tuple(df_SimilartyScore['cluster pair'][x])]=df_SimilartyScore['Similarity Score'][x]
#decending_pq=dict(zip(tuple(df_SimilartyScore['cluster pair']),df_SimilartyScore['Similarity Score']))
print(decending_pq)
orig_decendingpq=len(decending_pq)
first_in_que=0
next_cluster=0
while True:
#for next_cluster in range(0,len(df_SimilartyScore['cluster pair'])):
    print("beg.....",df_SimilartyScore['cluster pair'][next_cluster])
    ########## Appened to the node lit
    cluster1=int(df_SimilartyScore['cluster pair'][next_cluster][0])
    cluster2=int(df_SimilartyScore['cluster pair'][next_cluster][1])
    status=False
    print("merge node list ",merge_nodes_list)
    if cluster1 in merge_nodes_list or cluster2 in merge_nodes_list:
        print(decending_pq)
        for key,value in merge_nodes_dict.items():
            #print("key",key)
            for node in key:
                if cluster1==node:
                    cluster1 = value
                    status=True
                elif cluster2==node:
                    cluster2 = value
                    status=True
            if status==True:
                break;
        CUsers=0
        print("New cluster ",cluster1,cluster2)
        for nameC1 in ALLUserList[cluster1-1]:
            for nameC2 in ALLUserList[cluster2-1]:
                if nameC1==nameC2:
                    CUsers=CUsers+1
                    break;
        UserJd=CUsers/(len(ALLUserList[cluster1-1])+len(ALLUserList[cluster2-1])-CUsers)
        UserJdP=UserJd*100
        Jaccard_Threshold# print(Jaccard_Threshold)
        Userspr=0
        if UserJdP>Jaccard_Threshold:
        #    print(UserJdP)
            s1 = [];
            s2 = [];
            for p1 in range(0, len(ALLUserList[cluster1-1])):
                for p2 in range(0, len(ALLUserList[cluster2-1])):
                    if ALLUserList[cluster1-1][p1] == ALLUserList[cluster2-1][p2]:
                        s1.append(float(AllInfluenceScores[int(cluster1) - 1][p1]))
                        s2.append(float(AllInfluenceScores[int(cluster2) - 1][p2]))
                        #print(ALLUserList[cluster1-1][p1])
            if len(s1) > 1:
                Userspr=spearmans_rank_correlation(s1, s2)
            else:
                Userspr=0
        #print(Userspr)
        UsersimScore=Userspr*UserJd
        if UsersimScore>Similarity_Threshold:
           # print(UsersimScore)
            decending_pq[tuple([str(cluster1),str(cluster2)])]=UsersimScore
            decending_pq=dict(sorted(decending_pq.items(), key=operator.itemgetter(1), reverse=True))
            # print(decending_pq)
        if int(list(decending_pq.keys())[0][0])>14 or int(list(decending_pq.keys())[0][1])>14 :
            print(decending_pq)
            break
    else:
        print("in else")
        print(cluster1,cluster2)
        merge=[]
        merge.append(cluster1)
        merge.append(cluster2)
        merge_nodes_dict[tuple(merge)]=newCluster
        merge_nodes_list.append(cluster1)
        merge_nodes_list.append(cluster2)
        merge_nodes_list=list(sorted(set(merge_nodes_list)))
        ########## Merging the clusters
        first_cluster=updated_cluster[updated_cluster['clusters'] == int(df_SimilartyScore['cluster pair'][next_cluster][0])]
        second_cluster=updated_cluster[updated_cluster['clusters'] == int(df_SimilartyScore['cluster pair'][next_cluster][1])]
        merger_cluster=first_cluster.append(second_cluster, ignore_index=True)
        merger_cluster.loc[merger_cluster['clusters']== int(df_SimilartyScore['cluster pair'][next_cluster][0]) ,['clusters']]=newCluster
        merger_cluster.loc[merger_cluster['clusters']== int(df_SimilartyScore['cluster pair'][next_cluster][1]) ,['clusters']]=newCluster
        #print(merger_cluster)
        #print("---------------------------")
        aggregation_functions = {'clusters':'first','Weight_of_active_User':'first','UserName':'first','No.OfRetweets': 'sum','UserId': 'first','TweetId':'first', 'TweetCount': 'sum','TweetText':', '.join}
        merger_cluster=merger_cluster.groupby(['UserName']).aggregate(aggregation_functions)
        #print(merger_cluster)
        mergeclusters_data=updated_cluster.append(merger_cluster)
        #Drop the old clusters
        mergeclusters_data = mergeclusters_data.drop(mergeclusters_data[mergeclusters_data.clusters == cluster1].index)
        mergeclusters_data = mergeclusters_data.drop(mergeclusters_data[mergeclusters_data.clusters == cluster2].index)

        mergeclusters_data=mergeclusters_data.groupby(['clusters','UserName','UserId','TweetId','TweetText','Weight_of_active_User'])['No.OfRetweets','TweetCount'].sum().reset_index()
        mergeclusters_data['local_influential_score']=mergeclusters_data['No.OfRetweets']/mergeclusters_data['TweetCount']
        # mergeclusters_data=mergeclusters_data[mergeclusters_data['No.OfRetweets'] != cluster1]
        # mergeclusters_data=mergeclusters_data[mergeclusters_data['No.OfRetweets'] != cluster2]
        mergeclusters_data.to_csv("Merge_recluster.csv",index=False)
        new_Users=mergeclusters_data[mergeclusters_data['clusters']==newCluster]
        users=new_Users["UserName"].tolist()
        users_local_score=new_Users["local_influential_score"].tolist()
        #print(users)
        #print(len(users))
        #list_name = str(df["UserName"][i]).split(",")
        ALLUserList.append(users)
        AllInfluenceScores.append(users_local_score)
        updated_cluster=mergeclusters_data
        newCluster = newCluster + 1
        #break
    decending_pq.pop(list(decending_pq.keys())[0])
    sorted(decending_pq.items(),key=lambda item:item[1],reverse=True)
    print("Last step............")
    print(decending_pq)
    next_cluster=next_cluster+1
    if len(decending_pq)==0 or next_cluster>=orig_decendingpq:
        break

#Modify the updated merge cluster value (clusters) 1..n
print(mergeclusters_data["clusters"].unique())
old_clusters=mergeclusters_data["clusters"].unique().tolist()
clusters_count=len(mergeclusters_data["clusters"].unique().tolist())
IClusters_tweets_count=[]
for i in range(1,clusters_count+1):
    mergeclusters_data.loc[mergeclusters_data['clusters'] == mergeclusters_data["clusters"].unique()[i-1],['clusters']] = i
    IClusters_tweets_count.append(len(mergeclusters_data[mergeclusters_data["clusters"] ==i]))

print(mergeclusters_data["clusters"].unique())
#print(IClusters_tweets_count)

mergeclusters_data.to_csv("Merge_recluster.csv",index=False)

#We have to caluclate the Tweets Cluster Vector (TCV)

#TCV= {Sum, Wsum,cv,n, w,ftset,m}

final_data_vocab = pd.read_csv("final_users_sheet.csv",encoding='utf8')
print(len(final_data_vocab["TweetText"]))
#To caluclate sum
ub=0
lb=0
vocab=[];
for tweet in final_data_vocab["TweetText"].to_list():
    if ":" in tweet:
        tweet = tweet[tweet.index(":"):]

    tweet = re.sub(r"\r\n", " ", tweet)  # Remove multi line spcaces
    tweet = re.sub(r"[^ a-zA-Z]", " ", tweet)  # Remove all expcept mentioned
    tweet = re.sub(' +', ' ', tweet)  # Remove unneccarry white space between words
    i = i + 1
    StopWords = set(stopwords.words("english"))
    rem_stopWords=["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]
    tweet_noSW = ' '.join([word.lower() for word in tweet.split() if word.lower() not in StopWords])  # Stopwords removal
    #Filters letters less than 3
    tweet_noSW = ' '.join([word.lower() for word in tweet_noSW.split() if len(word)>=3])
    word_list = words.words()
    tweet_noSW = ' '.join([word for word in tweet_noSW.split() if word in word_list])
    print(i, "-->", tweet_noSW)
    ps = PorterStemmer()

    tweet_words = list(OrderedDict.fromkeys(word_tokenize(tweet_noSW)))
    for w in tweet_words:
        vocab.append(ps.stem(w))

Matrix_col=len(set(vocab))
print("Total vocab length ",Matrix_col)
vocab=list(OrderedDict.fromkeys(vocab))      #This is for total vocabulary calucaltion

totalCvlist=[]
totalSimilarity_htest=[]
for clusterTotCount in IClusters_tweets_count:
    i=0
    ub=ub+clusterTotCount
    trim_tweets = []
    #print("cluster ", lb, ub)
    for tweet in mergeclusters_data["TweetText"].to_list()[lb:ub]:
        if ":" in tweet:
            tweet = tweet[tweet.index(":"):]

        tweet = re.sub(r"\r\n", " ", tweet)  # Remove multi line spcaces
        tweet = re.sub(r"[^ a-zA-Z]", " ", tweet)  # Remove all expcept mentioned
        tweet = re.sub(' +', ' ', tweet)  # Remove unneccarry white space between words
        i = i + 1
        StopWords = set(stopwords.words("english"))
        rem_stopWords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any",
                         "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
                         "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't",
                         "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few",
                         "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
                         "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
                         "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll",
                         "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself",
                         "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or",
                         "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't",
                         "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t",
                         "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",
                         "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
                         "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which",
                         "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y",
                         "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
                         "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's",
                         "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've",
                         "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]
        tweet_noSW = ' '.join(
            [word.lower() for word in tweet.split() if word.lower() not in StopWords])  # Stopwords removal
        # Filters letters less than 3
        tweet_noSW = ' '.join([word.lower() for word in tweet_noSW.split() if len(word) >= 3])
        word_list = words.words()
        tweet_noSW = ' '.join([word for word in tweet_noSW.split() if word in word_list])
        #print(i, "-->", tweet_noSW)
        ps = PorterStemmer()

        tweet_words = list(OrderedDict.fromkeys(word_tokenize(tweet_noSW)))
        # for w in tweet_words:
        #     print(ps.stem(w), end=",")  # Stemming
        #     vocab.append(ps.stem(w))
        tweetTextStem = ' '.join([ps.stem(word) for word in tweet_words])
        #print("\n")
        #print(tweetTextStem)
        trim_tweets.append(tweetTextStem)
        #print("\n")
    # Matrix_col = len(set(vocab))
    # print(" ", Matrix_col)
    # vocab = list(OrderedDict.fromkeys(vocab))
    tf = TfidfVectorizer(vocabulary=vocab)
    text_tfidf = tf.fit_transform(trim_tweets)
    # text_tf=text_tf.todense()
    text_tf = text_tfidf.toarray()
    #print(text_tf)
    print("----")

    #print(len(text_tf[0]))
    distance_array = [];
    print("------")
    for i in range(0, len(text_tf)):
        distance = 0
        for val in text_tf[i]:
            if val != 0:
                distance = distance + val * val
        distance = math.sqrt(distance)
        distance_array.append(distance)

    #print(distance_array)
    cluster_i=1
    print("Results Cluster ",cluster_i)
    sumCalList=[]
    for index in range(0,len(text_tf[0])):
        sumCal=0
        for tweetnum in range(0,len(distance_array)):
            if distance_array[tweetnum]==0:
                sumCal=sumCal
            else:
                sumCal = sumCal + (text_tf[tweetnum][index] / distance_array[tweetnum])
        sumCalList.append(sumCal)
    print("Sum :",sumCalList)
    wSumList=[]#df["Weight_of_active_User"]=numpy.log2(df['No.OfRetweets'])
    #mergeclusters_data["Weight_of_active_User"]
    for index in range(0,len(text_tf[0])):
        wSumCal=0
        for tweetnum in range(0,len(distance_array)):
            wSumCal=wSumCal+(text_tf[tweetnum][index]*mergeclusters_data["Weight_of_active_User"][tweetnum])
        wSumList.append(wSumCal)
    print("wSum :",wSumList)
    cvList=[] #cv=wsum/sigma w
    sig_w=0
    for weight in mergeclusters_data["Weight_of_active_User"].tolist()[lb:ub]:
        sig_w=sig_w+weight
    Mod_cv = 0
    for wsum in wSumList:
        val=wsum/sig_w
        cvList.append(val)
        Mod_cv=Mod_cv+(val*val)
    print("cvList  :",cvList)
    print("cvList percentage :",Mod_cv)
    N=ub-lb
    print("total number of Tweets (N) :",N)

    _Similarity_list=[] #text_tf[0] and cvList
    _similarity=0
    for text_word_idx in range(0,len(text_tf)):
        for _wordtfidf in range(0,len(text_tf[text_word_idx])):
            _similarity=_similarity+text_tf[text_word_idx][_wordtfidf]*cvList[_wordtfidf]
        _similarity=(_similarity)/(distance_array[i]*Mod_cv)
        _Similarity_list.append(_similarity)
        _similarity=0

    _Similarity_htest=sorted(_Similarity_list,reverse=True)
    print("Similarity Score:",_Similarity_htest)
    totalSimilarity_htest.append(_Similarity_htest)
    #print(text_tf[1])
    totalCvlist.append(cvList)
    lb=ub+1
    cluster_i=cluster_i+1

old_tot_cls=len(totalSimilarity_htest)
#Inceement cluster began--------------------------
Quartertweets=int(len(final_data_vocab["TweetText"])/4)
remTweets_data=final_data_vocab[Quartertweets:]

print("************")
print("Total tweets After K means Clustering :",len(final_data_vocab["TweetText"])-Quartertweets)
#Now we have to divide agian these into Four Halves

N_IClusters_tweets_count=[]
lb=0
ub=int(len(remTweets_data["TweetText"])/4)
incre=ub
table_val = [clusters_count, "--", "--", "--"]
final_output.append(table_val)
_newtweet=len(mergeclusters_data)+1
for parts in range(0,4):
    trim_tweets = []
    for tweet in remTweets_data["TweetText"].to_list()[lb:ub]:
        if ":" in tweet:
            tweet = tweet[tweet.index(":"):]

        tweet = re.sub(r"\r\n", " ", tweet)  # Remove multi line spcaces
        tweet = re.sub(r"[^ a-zA-Z]", " ", tweet)  # Remove all expcept mentioned
        tweet = re.sub(' +', ' ', tweet)  # Remove unneccarry white space between words
        i = i + 1
        StopWords = set(stopwords.words("english"))
        rem_stopWords=["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"]
        tweet_noSW = ' '.join([word.lower() for word in tweet.split() if word.lower() not in StopWords])  # Stopwords removal
        tweet_noSW = ' '.join([word.lower() for word in tweet_noSW.split() if len(word)>=3])
        word_list = words.words()
        tweet_noSW = ' '.join([word for word in tweet_noSW.split() if word in word_list])
        print(i, "-->", tweet_noSW)
        ps = PorterStemmer()

        tweet_words = list(OrderedDict.fromkeys(word_tokenize(tweet_noSW)))
        tweetTextStem=' '.join([ps.stem(word) for word in tweet_words])
        trim_tweets.append(tweetTextStem)

    tf = TfidfVectorizer(vocabulary=vocab)
    text_tfidf = tf.fit_transform(trim_tweets)

    remtext_tf=text_tfidf.toarray()
    print(remtext_tf)

    print(len(remtext_tf[0]))
    tfidfdistance_array=[];
    for i in range(0,len(remtext_tf)):
        distance=0
        for val in remtext_tf[i]:
            if val!=0:
                distance=distance+val*val
        distance=math.sqrt(distance)
        tfidfdistance_array.append(distance)

    print(tfidfdistance_array)
    print(totalCvlist)
    print("total cv list......",len(totalCvlist))
    Mag_cv=[]
    for cvlist in totalCvlist:
        sum=0
        for cv in cvlist:
            sum=sum+(cv*cv)
        sum=math.sqrt(sum)
        Mag_cv.append(sum)

    #print("Lengths....... ",len(remtext_tf[0]),len(totalCvlist[0]))
    # _newtweet=len(remtext_tf)
   # print("*********",len(remtext_tf))
    _distanceArray=[]
    inital_cv_lenth=len(totalCvlist)
    for tweetnum in  range(0,len(remtext_tf)):
        N_Similarity_list=[]
        _i = 0
        cluster_index = []
        #print("Length ",len(totalCvlist))
        for cvlist in totalCvlist:
            _similarity=0
            mgCv=0
            cluster_index.append(_i+1)
            for index in range(0,len(cvlist)):
                _similarity=_similarity+remtext_tf[tweetnum][index]*cvlist[index]
                mgCv = mgCv + (cvlist[index] * cvlist[index])
                mgCv = math.sqrt(sum)
            if (tfidfdistance_array[tweetnum] * mgCv)==0:
                _similarity = (_similarity)
            else:
                _similarity = (_similarity) / (tfidfdistance_array[tweetnum] * mgCv)
            N_Similarity_list.append(_similarity)
            _similarity=0
            _i=_i+1

        c_w_ss = dict(zip(N_Similarity_list, cluster_index))
        _c_w_ss = sorted(c_w_ss.keys(), reverse=True)
        C_w_ma_ss = c_w_ss[_c_w_ss[0]]
        C_w_ma_skey = _c_w_ss[0]
        #print(_c_w_ss)
        #print(C_w_ma_skey,"------->",C_w_ma_ss)
        #Find MBS
        MBS=0
        thresh=0.05 #Threshold Value
        _avg=0.0
        #print(len(totalSimilarity_htest))
        #print("Htest ",totalSimilarity_htest[C_w_ma_ss-1])
        neList = []
        for val in totalSimilarity_htest[C_w_ma_ss-1]:
            if type(val) is not list:
                neList.append(val)
        totalSimilarity_htest[C_w_ma_ss-1]=neList
        for c_ss in totalSimilarity_htest[C_w_ma_ss-1]:
           if str(c_ss)!='inf':
                _avg=_avg+c_ss
        if _avg>0:
           # print(_avg)
            _avg=_avg/len(totalSimilarity_htest[C_w_ma_ss-1])
        else:
            _avg=0

        MBS=thresh*_avg
       # print("MBS--->",MBS)
        if _newtweet>=len(final_data_vocab['TweetText']):
            break
        if C_w_ma_skey<MBS:
            #New clusters
            #remTweets_data.iloc[_newtweet]
            old_tot_cls=old_tot_cls+1
            print("Tweet Number ",_newtweet)
            row=[
                 old_tot_cls,
                 final_data_vocab.iloc[_newtweet]['UserName'],
                 final_data_vocab.iloc[_newtweet]['UserId'],
                 final_data_vocab.iloc[_newtweet]['TweetId'],
                 final_data_vocab.iloc[_newtweet]['TweetText'],
                 final_data_vocab.loc[_newtweet]['Weight_of_active_User'],
                 final_data_vocab.iloc[_newtweet]['No.OfRetweets'],
                 final_data_vocab.iloc[_newtweet]['TweetCount'],
                 "***"
            ]
            mergeclusters_data.loc[_newtweet] = row
            #mergeclusters_data=mergeclusters_data.append(row)
            if len(mergeclusters_data[mergeclusters_data["clusters"] == old_tot_cls])==1:
                totalCvlist.append(remtext_tf[tweetnum])
                n_cvlist=remtext_tf[tweetnum]
                __similarity=[]
                for v in range(0,len(n_cvlist)):
                    __similarity.append(i)
                _Similarity_list.append(__similarity)
    #            _Similarity_htest = sorted(_Similarity_list, reverse=True)
               # print("Similarity Score:", _Similarity_list)
                totalSimilarity_htest.append(_Similarity_list)

        else:
            #Merge with cp
            row = [
                C_w_ma_ss,
                final_data_vocab.iloc[_newtweet]['UserName'],
                final_data_vocab.iloc[_newtweet]['UserId'],
                final_data_vocab.iloc[_newtweet]['TweetId'],
                final_data_vocab.iloc[_newtweet]['TweetText'],
                final_data_vocab.loc[_newtweet]['Weight_of_active_User'],
                final_data_vocab.iloc[_newtweet]['No.OfRetweets'],
                final_data_vocab.iloc[_newtweet]['TweetCount'],
                "***"
            ]
            mergeclusters_data.loc[_newtweet] = row
            #mergeclusters_data=mergeclusters_data.append(row)
            trim_tweets=[]
            if C_w_ma_ss>inital_cv_lenth:
             #   print("In more than 1 loop..........")
                #New Cluster
                NewCv=[]
                C_WTweets=mergeclusters_data[mergeclusters_data["clusters"] == C_w_ma_ss]
               # print(C_WTweets)
                #print("----------len",len(C_WTweets))
                #print(C_WTweets["TweetText"].to_list())
                for tweet in C_WTweets["TweetText"].to_list():
                    if ":" in tweet:
                        tweet = tweet[tweet.index(":"):]

                    tweet = re.sub(r"\r\n", " ", tweet)  # Remove multi line spcaces
                    tweet = re.sub(r"[^ a-zA-Z]", " ", tweet)  # Remove all expcept mentioned
                    tweet = re.sub(' +', ' ', tweet)  # Remove unneccarry white space between words
                    i = i + 1
                    StopWords = set(stopwords.words("english"))
                    rem_stopWords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any",
                                     "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below",
                                     "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't",
                                     "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few",
                                     "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have",
                                     "haven",
                                     "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
                                     "how",
                                     "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll",
                                     "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself",
                                     "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only",
                                     "or",
                                     "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan",
                                     "shan't",
                                     "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t",
                                     "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there",
                                     "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve",
                                     "very",
                                     "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where",
                                     "which",
                                     "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y",
                                     "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
                                     "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's",
                                     "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've",
                                     "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's",
                                     "would"]
                    tweet_noSW = ' '.join(
                        [word.lower() for word in tweet.split() if word.lower() not in StopWords])  # Stopwords removal
                    # Filters letters less than 3
                    tweet_noSW = ' '.join([word.lower() for word in tweet_noSW.split() if len(word) >= 3])
                    word_list = words.words()
                    tweet_noSW = ' '.join([word for word in tweet_noSW.split() if word in word_list])
                    print(i, "-->", tweet_noSW)
                    ps = PorterStemmer()
                    tweet_words = list(OrderedDict.fromkeys(word_tokenize(tweet_noSW)))
                    tweetTextStem = ' '.join([ps.stem(word) for word in tweet_words])
                    trim_tweets.append(tweetTextStem)

                tf = TfidfVectorizer(vocabulary=vocab)
                Newtext_tfidf = tf.fit_transform(trim_tweets)
                # text_tf=text_tf.todense()
                newtext_tf = Newtext_tfidf.toarray()
                # print(text_tf)
                #print(newtext_tf)

                Newdistance_array = [];
                #print("------")
                for i in range(0, len(newtext_tf)):
                    _distance = 0
                    for val in newtext_tf[i]:
                        if val != 0:
                            _distance = _distance + val * val
                    _distance = math.sqrt(_distance)
                    Newdistance_array.append(_distance)

                #print("Results Cluster ", C_w_ma_ss)
                sumCalList = []
                print(len(newtext_tf[0]))
                print(newtext_tf[0])
                for index in range(0, len(newtext_tf[0])):
                    sumCal = 0
                   # print(len(newtext_tf))
                   # print(len(Newdistance_array))
                    for tweet_num in range(0, len(Newdistance_array)):
                        if Newdistance_array[tweet_num] != 0:
                           # print(newtext_tf[tweet_num])
                            sumCal = sumCal + (newtext_tf[tweet_num][index] / Newdistance_array[tweet_num])
                    sumCalList.append(sumCal)
                #print("Sum :", sumCalList)
                #print(C_WTweets["Weight_of_active_User"])
                wSumList = []  # df["Weight_of_active_User"]=numpy.log2(df['No.OfRetweets'])
                # mergeclusters_data["Weight_of_active_User"]
                for index in range(0, len(newtext_tf[0])):
                    wSumCal = 0
                    for tweet_num in range(0, len(Newdistance_array)):
                        wSumCal = wSumCal + (newtext_tf[tweet_num][index] * C_WTweets['Weight_of_active_User'].to_list()[tweet_num])
                    wSumList.append(wSumCal)
                #print("wSum :", wSumList)
                cvList = []  # cv=wsum/sigma w
                sig_w = 0
                for weight in C_WTweets["Weight_of_active_User"].tolist():
                    sig_w = sig_w + weight
                Mod_cv = 0
                for wsum in wSumList:
                    val = wsum / sig_w
                    cvList.append(val)
                    Mod_cv = Mod_cv + (val * val)
                #print("cvList  :", cvList)
                #print("cvList percentage :", Mod_cv)
                _similarity = 0
                for text_word_idx in range(0, len(newtext_tf)):
                    for _wordtfidf in range(0, len(newtext_tf[text_word_idx])):
                        _similarity = _similarity + newtext_tf[text_word_idx][_wordtfidf] * cvList[_wordtfidf]
                    if Newdistance_array[text_word_idx]!=0:
                       similarity = (_similarity) / (Newdistance_array[text_word_idx] * Mod_cv)
                _Similarity_list.append(_similarity)
                _similarity = 0
                #print("Similarity Score:", _Similarity_list)
                if C_w_ma_ss==len(totalCvlist):
                    totalSimilarity_htest.append(_Similarity_list)
                    totalCvlist.append(cvList)
                else:
                    totalSimilarity_htest[C_w_ma_ss] = _Similarity_list
                    totalCvlist[C_w_ma_ss] = cvList
                #print("Length of total cv list", len(totalCvlist))

        _newtweet=_newtweet+1

    lb=ub
    ub=ub+incre

    mergeclusters_data = mergeclusters_data.sort_values('clusters')
    mergeclusters_data.to_csv("Final_lastCLusters.csv",index=False)   #Contains newly formed clusters
    print(mergeclusters_data["clusters"].unique())
    final_clusters_count=len(mergeclusters_data["clusters"].unique().tolist())  #Total Updated clusters count
    final_updatedclusters_list=mergeclusters_data["clusters"].unique().tolist()
    O_IClusters_tweets_count=[]
    for i in range(1,clusters_count+1):
        O_IClusters_tweets_count.append(len(mergeclusters_data[mergeclusters_data["clusters"] ==i]))
    print("For Old clusters")
    O_G_cluster=[]
    GC_count=0
    OD_count=0
    for val in range(0,len(IClusters_tweets_count)):
        newNess=O_IClusters_tweets_count[val]/IClusters_tweets_count[val]
        if newNess>1:
            GC_count=GC_count+1
            O_G_cluster.append([old_clusters[val],(O_IClusters_tweets_count[val]+IClusters_tweets_count[val]),newNess,"Growing Clutser"])
        else:
            OD_count=OD_count+1
            O_G_cluster.append([old_clusters[val], (O_IClusters_tweets_count[val] + IClusters_tweets_count[val]),newNess,"Outdated Clutser"])
    for o_c in O_G_cluster:
        print(o_c)

    print("For New Clusters")
    N_G_cluster=[]
    N_IClusters_tweets_count=[]
    for i in range(len(IClusters_tweets_count),final_clusters_count):
        N_IClusters_tweets_count.append(len(mergeclusters_data[mergeclusters_data["clusters"] ==i]))

    #print(len(N_IClusters_tweets_count))
    IF_count=0
    if parts==0:
        for val in range(clusters_count+1,final_clusters_count):
            IF_count=IF_count+1
            N_G_cluster.append([final_updatedclusters_list[val],'1.0',"infant CLuster"])
        for n_c in N_G_cluster:
            print(n_c)
    else:
        i=0
        #val=len(IClusters_tweets_count) + 1
        #for val in range(clusters_count+1,final_clusters_count): O_IClusters_tweets_count
        beta=0.5
        alpha=0.1
        for i_len in range(len(IClusters_tweets_count)+1,clusters_count):
            try:
                n=Old_N_checkClusters_count[i_len-len(IClusters_tweets_count)] /N_IClusters_tweets_count[i_len]
                if (N_IClusters_tweets_count[i_len]-(Old_N_checkClusters_count[i_len - len(IClusters_tweets_count)])) < (beta*(incre / GC_count)):  #Fomula need to change
                    GC_count=GC_count+1
                    N_G_cluster.append([final_updatedclusters_list[i_len],n,"Growing CLuster"])
                else:
                    #Alpha activity
                    if (N_IClusters_tweets_count[i_len]-(Old_N_checkClusters_count[i_len - len(IClusters_tweets_count)])) < (alpha*(incre / GC_count)):  #Fomula need to change
                        IF_count = IF_count + 1
                        N_G_cluster.append([final_updatedclusters_list[i_len], n, "Infant CLuster"])
                    else:
                        OD_count = OD_count + 1
                        N_G_cluster.append([final_updatedclusters_list[i_len], n, "Outdated CLuster"])
               # val=val+1
            except:
                GC_count=GC_count+1
                N_G_cluster.append([final_updatedclusters_list[i_len], n, "Growing CLuster"])
                print("Neglect")
                #i=i+1
        for val in range(clusters_count+1,final_clusters_count):
            IF_count=IF_count+1
            N_G_cluster.append([final_updatedclusters_list[val],'1.0',"infant CLuster"])
        for n_c in N_G_cluster:
            print(n_c)
    Old_N_checkClusters_count = N_IClusters_tweets_count
    clusters_count=final_clusters_count
   # IClusters_tweets_count=O_IClusters_tweets_count
    table_val=["--",GC_count,OD_count,IF_count]
    final_output.append(table_val)

print("********Final Output*******")
print(final_output)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
#y = [1, 2, 3, 4, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1]
col_labels = ('Clusters', 'Growing Cluster', 'Outdated Cluster', 'Infant Cluster')
row_labels=["Initial","1st Part","2nd Part","3rd Part","4th part"]

# Draw table
the_table = plt.table(cellText=final_output,
                      colWidths=[0.1] * 4,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center')
the_table.auto_set_font_size(False)
the_table.set_fontsize(10)
the_table.scale(2.5, 3)

# Removing ticks and spines enables you to get the figure only with table
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
for pos in ['right','top','bottom','left']:
    plt.gca().spines[pos].set_visible(False)
plt.savefig('matplotlib-table_beta03_alpha80.png', bbox_inches='tight', pad_inches=0.05)

plt.show()


