---
title: "Datavis 4"
author: "luqi Chen"
date: "4/11/2019"
output:
  html_document:
    keep_md: true
---



##1.Who follows whom?

a) Network of Followers


```r
followdata = read.csv("senators_follow.csv")
followdata_true <- followdata[which(followdata$following=='TRUE'),]
followdata_true_st = followdata_true[,c("source","target")]

library(igraph)
# create the network object
set.seed(10)
network_1=graph_from_data_frame(d=followdata_true_st, directed=T) 
# plot it
plot(network_1)
```

![](Datavis_4_files/figure-html/unnamed-chunk-1-1.png)<!-- -->

```r
library(dplyr)
most_followers <- followdata_true_st %>%
  dplyr::group_by(target) %>%
  dplyr::summarise(n_followers = n()) %>%
  top_n(3)
```

```
## Selecting by n_followers
```

```r
most_followers
```

```
## # A tibble: 4 x 2
##   target          n_followers
##   <fct>                 <int>
## 1 MarkWarner               94
## 2 SenatorCantwell          95
## 3 SenJohnKennedy           95
## 4 SenSherrodBrown          94
```

```r
followmost <- followdata_true_st %>%
  dplyr::group_by(source) %>%
  dplyr::summarise(n_follow = n()) %>%
  top_n(3)
```

```
## Selecting by n_follow
```

```r
followmost
```

```
## # A tibble: 4 x 2
##   source         n_follow
##   <fct>             <int>
## 1 ChuckGrassley        78
## 2 lisamurkowski        78
## 3 SenatorCollins       85
## 4 SenShelby           119
```

```r
# indegree = igraph::degree(network1,mode="in")
# indegree_df = as.data.frame(indegree)
# names_df <- followdata_true_st %>%
#   group_by(source)
# indegree_df_top3 <- indegree_df %>%
#   arrange(desc(indegree)) %>%
#   head(3)
# row.names(indegree_df_top3)
```
The four senators who are followed by the most of their colleagues:
SenatorCantwell(95),SenJohnKennedy(95),MarkWarner(94),SenSherrodBrown(94)

The four senators who follow the most of their colleagues:
ChuckGrassley(119),SenatorCollins(85),lisamurkowski(78),ChuckGrassley(78)

```r
senatordata = read.csv("senators_twitter.csv")
network_2 <- graph_from_data_frame(followdata_true_st, directed=FALSE)
V(network_2)$size <- centralization.degree(network_2)$res
E(network_2)$weight <- 1
g1 <- igraph::simplify(network_2, edge.attr.comb="sum")
library(intergraph)
set.seed(2103)
plotdata1 <- ggnetwork(g1, layout="fruchtermanreingold", 
          arrow.gap=0, cell.jitter=0)
# merge it with party information
plotdata2 <- merge(x = plotdata1, y = senatordata, by.x = "vertex.names", by.y = "Official_Twitter", all = TRUE)

#fulltable= merge(x = followdata, y = senatordata, by.x = "source", by.y = "Official_Twitter", all = TRUE)

# plot the networkgraph
library(ggplot2)
library(ggrepel)
(networkg1 <- ggplot() +
  geom_edges(data=plotdata2, 
             aes(x=x, y=y, xend=xend, yend=yend),
             color="grey50", curvature=0.1, size=0.15, alpha=1/2) +
  geom_nodes(data=plotdata2,
             aes(x=x, y=y, xend=xend, yend=yend,size = sqrt(size),color = Party_affiliation)) +   
  geom_label_repel(data=unique(plotdata2[plotdata2$size>150,c(1,2,3)]),
                   aes(x=x, y=y, label=vertex.names), 
                   size=2, color="#8856a7") +
  theme_blank() +
  theme(legend.position="none") + scale_colour_manual(values=c("#0072B2", "#009E73","#D55E00")))
```

```
## Warning: Ignoring unknown aesthetics: xend, yend
```

```
## Warning: Removed 7 rows containing missing values (geom_point).
```

```
## Warning: Removed 1 rows containing missing values (geom_label_repel).
```

![](Datavis_4_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

b) Communities


```r
wc <- cluster_walktrap(g1)
members = membership(wc)
#members
com <- cbind(V(g1)$name,wc$membership)
cluster_DF <- as.data.frame(com)
colnames(cluster_DF) = c("Official_Twitter", "party")
cluster_DF$classification_method = c("community_detection")
senator_DF <- senatordata[,c("Official_Twitter","Party_affiliation")]
senator_DF$classification_method = c("true_party")
senator_DF$party <- ifelse(senator_DF$Party_affiliation=="Republican Party",1,2)
colnames(senator_DF) = c("Official_Twitter", "Party_affiliation","classification_method","party")
senator_DF2 <- senator_DF[,c("Official_Twitter","party","classification_method")]
data_1b <- rbind(cluster_DF,senator_DF2)
data_1b_group <- data_1b %>%
  dplyr::group_by(Official_Twitter, classification_method) %>%
  arrange(desc(Official_Twitter))
data1b_first50 = data_1b_group[1:101,]
data1b_next50 = data_1b_group[102:199,]
pointg1_first50 = ggplot(data1b_first50, aes(x=Official_Twitter, y=party, shape=classification_method, color=classification_method))
pointg1_first50 + geom_point() + coord_flip() + scale_shape_manual(values=c(0, 4)) + scale_colour_manual(values=c("#0072B2", "#D55E00"))
```

![](Datavis_4_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

```r
pointg1_next50 = ggplot(data1b_next50, aes(x=Official_Twitter, y=party, shape=classification_method, color=classification_method))
pointg1_next50 + geom_point() + coord_flip() + scale_shape_manual(values=c(0, 4)) + scale_colour_manual(values=c("#0072B2", "#D55E00"))
```

![](Datavis_4_files/figure-html/unnamed-chunk-3-2.png)<!-- -->

note: party is "republic" when it takes value "1" and is "democratic" when it takes value "2".The blue square means the value of party is given by my community detection data, the orange cross means the value of party is the true party affiliation of the senator.
To view every line clearly, I seperated my data 2 subsets, each has 50 senators.

Comment: My graphs show that the automated community detection mechanism recovers all the party affliliation of senators.(some of those rows only have square or only have cross, this means there is information blank in one of the csv sheets, this does not affect my conclusion that they recover well with each other)

##2.What are they tweeting about?

a) Most Common Topics Over Time


```r
st = readRDS("senator_tweets.RDS", refhook = NULL)
st_original <- st[which(st$is_retweet=='FALSE'),]
st_original_value <- st_original[!is.na(st_original$hashtags), ]
st_original_hash <- st_original_value[,"hashtags"]
ID89888 = as.data.frame(c(1:89888))
hashtext = cbind(ID89888,st_original_hash)
colnames(hashtext) = c("doc_id", "text")
library(tm)
```

```
## Loading required package: NLP
```

```
## 
## Attaching package: 'NLP'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     annotate
```

```r
library(tidytext)
hash_source = DataframeSource(hashtext)
hash_corpus <- VCorpus(hash_source)
removeNumPunct <- function(x){gsub("[^[:alpha:][:space:]]*", "", x)}
clean_corpus_2 <- function(corpus){
  corpus <- tm_map(corpus, removeWords, c(stopwords("en")))  
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(removeNumPunct))
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}
hash_clean <- clean_corpus_2(hash_corpus)
hash_TDM2 = TermDocumentMatrix(hash_clean)
hash_td2 = tidy(hash_TDM2)
library(dplyr)
library(tidytext)
hash_Top10 <- hash_td2 %>%
  dplyr::group_by(term) %>%
  dplyr::summarise(total = sum(count)) %>%
  arrange(desc(total)) %>%
  top_n(10) 
```

```
## Selecting by total
```

```r
g_hash = ggplot(data=hash_Top10,aes(x=reorder(term,total),y=total))
g_hash + geom_bar(stat = "identity",fill ="#009E73")+coord_flip()+ggtitle("Most Common Topics Over Time")+labs(x="hashtags", y="count") +theme(legend.position = "top")
```

![](Datavis_4_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

b) Russia investigation - Dems vs. Reps


```r
MuellerSubset <- st[grep("Mueller", st$hashtags), ]
# MuellerSubset[56,5]
# MuellerSubset2 <- st[grep("Mueller", st$text), ]
# MuellerSubset2_value <- MuellerSubset2[!is.na(MuellerSubset2$hashtags), ]
# Agentsubset <- st[grep("Agent", st$hashtags), ]
# Sessionsubset <- st[grep("Sessions", st$hashtags), ]
# Subset3 <- st[grep("investigation", st$text), ]
# Subset4 <- st[grep("Counsel", st$text), ]
subset5 <- st[grep("Interference", st$hashtags), ]
```
First Step: identify support and critical
from above we can see that the hashtag "ProtectMueller" also signals support for Robert Muellers work
So the 5 hashtags that signal support are: #ProtectMueller, #MuellerInvestigation, #MuellerReport, #MuellerYoureListening and #RobertMueller
we can also wee that the hashtag "RussiaInterference" and "Putin" signal critical sentiment towards the investigation, so the 5 hashtags that signal critical sentiment are: #WitchHunt, #fakenews, #NoCollusion, #RussianInterference, #Putin

```r
st_value <- st[!is.na(st$hashtags), ]
st_hash <- st_value[,"hashtags"]
ID103786 = as.data.frame(c(1:103786))
hashtext2 = cbind(ID103786,st_hash)
colnames(hashtext2) = c("doc_id", "text")
library(tm)
library(tidytext)
hash_source2 = DataframeSource(hashtext2)
hash_corpus2 <- VCorpus(hash_source2)
hash_clean2 <- clean_corpus_2(hash_corpus2)
hash_TDM3 = TermDocumentMatrix(hash_clean2)
hash_td3 = tidy(hash_TDM3)
# do a subset of the hashtags that only contain the 10 hashtags 
subset2b <- hash_td3[which(hash_td3$term %in% c("protectmueller","muellerinvestigation","muellerreport","muelleryourelistening","robertmueller","witchhunt","fakenews","nocollusion","russianinterference","putin")),]
st_name = st_value[,"screen_name"]
st_name $ document = c(1:103786)
st_name$document_name <- paste("document", "-", st_name$document)
subset2b$document_name<- paste("document", "-", subset2b$document)
subset2b_2= dplyr::left_join(subset2b,st_name,by="document_name")
colnames(subset2b_2) = c("term", "id1","count","document_name","Official_Twitter","id2")
subset2b_3 = dplyr::left_join(subset2b_2,senator_DF,by="Official_Twitter")
```

```
## Warning: Column `Official_Twitter` joining character vector and factor,
## coercing into character vector
```

```r
subset2b_4 = subset2b_3[,c("term","count","Official_Twitter","Party_affiliation")]
subset2b_4_value = drop_na(subset2b_4)
subset2b_4_group <- subset2b_4_value %>%
  dplyr::group_by(term,Party_affiliation) %>%
  dplyr::summarise(total = sum(count)) %>%
  arrange(desc(total))
g_mueller = ggplot(data=subset2b_4_group,aes(x=reorder(term,total),y=total,fill=Party_affiliation))
g_mueller + geom_bar(stat = "identity")+scale_fill_manual(values=c("#0072B2", "#D55E00")) + ggtitle("use of hashtags related to mueller") + labs (x="hashtag",y="count") 
```

![](Datavis_4_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

From the graph we can see that for the hashtags that signal support for Mueller(muellerinvestigation, muellerreport and protectmueller), there are more democratic senators using them; for the hashtags that signal critical sentiment for Mueller(fakenews, putin),there are more republic senators using them.

c) Russia investigation - Barr report


```r
Subset6 <- st[grep("Barr", st$text), ]
# Set all the data about Barr that is created after March 24
Subset7 <- Subset6 %>%
  arrange(desc(created_at))%>%
  head(92)
# first let's see the party affiliation of the senators who commented on the event
Subset7_name <- Subset7[,"screen_name"]
colnames(Subset7_name) <- c("Official_Twitter")
subset8 = dplyr::left_join(Subset7_name,senator_DF,by="Official_Twitter")
```

```
## Warning: Column `Official_Twitter` joining character vector and factor,
## coercing into character vector
```

```r
subset8 = drop_na(subset8)
g_Barr1 = ggplot(subset8, aes(x=factor(1), fill=Party_affiliation))+
  geom_bar(width = 1)+
  coord_polar("y") + labs (x="",y="") + ggtitle("Tweets Counts Responding to Barr Report")
g_Barr1 + scale_fill_manual(values=c("#0072B2","#009E73","#D55E00"))
```

![](Datavis_4_files/figure-html/unnamed-chunk-7-1.png)<!-- -->


```r
# Second lets do a sentiment analysis of the tweets contents of Democratic Party and Republican Party respectively and then compare them
subset9 <- Subset7[,c("screen_name","text")]
colnames(subset9) <- c("Official_Twitter","text")
subset10 <- dplyr::left_join(subset9,senator_DF,by="Official_Twitter")
```

```
## Warning: Column `Official_Twitter` joining character vector and factor,
## coercing into character vector
```

```r
subset10_democratic <- subset10[which(subset10$Party_affiliation=='Democratic Party'),]
subset10_republican <- subset10[which(subset10$Party_affiliation=='Republican Party'),]
demo <- subset10_democratic[,"text"]
repub <- subset10_republican[,"text"]
ID71 <- as.data.frame(c(1:71))
ID17 <- as.data.frame(c(1:17))
textdemo <- cbind(ID71,demo)
textrepub <- cbind(ID17,repub)
colnames(textdemo) = c("doc_id", "text")
colnames(textrepub) = c("doc_id", "text")
demo_source = DataframeSource(textdemo)
repub_source = DataframeSource(textrepub)
demo_corpus <- VCorpus(demo_source)
repub_corpus <- VCorpus(repub_source)
# Define clean_corpus function
clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en")))  
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(removeNumPunct))
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}
demo_clean <- clean_corpus(demo_corpus)
repub_clean <- clean_corpus(repub_corpus)
library(SnowballC)
```

```
## Warning: package 'SnowballC' was built under R version 3.5.2
```

```r
demo_stemmed = tm_map(demo_clean, stemDocument)
repub_stemmed = tm_map(repub_clean, stemDocument)
# Complete the stems
stemCompletion2 <- function(x, dictionary) {
   x <- unlist(strsplit(as.character(x), " "))
    # # Oddly, stemCompletion completes an empty string to
	  # a word in dictionary. Remove empty string to avoid issue.
   x <- x[x != ""]
   x <- stemCompletion(x, dictionary=dictionary)
   x <- paste(x, sep="", collapse=" ")
   PlainTextDocument(stripWhitespace(x))
}
library(parallel)
library(pbapply) 
```

```
## Warning: package 'pbapply' was built under R version 3.5.2
```

```r
# Calculate the number of cores
no_cores <- detectCores() - 1
# Initiate cluster
demo_comp_all <- mclapply(demo_stemmed, 
                          stemCompletion2, 
                          dictionary = demo_clean, 
                          mc.cores=no_cores)
demo_comp_all <- as.VCorpus(demo_comp_all)
demo_tdm <- TermDocumentMatrix(demo_comp_all)
demo_td <- tidy(demo_tdm)

repub_comp_all <- mclapply(repub_stemmed, 
                          stemCompletion2, 
                          dictionary = repub_clean, 
                          mc.cores=no_cores)
repub_comp_all <- as.VCorpus(repub_comp_all)
repub_tdm <- TermDocumentMatrix(repub_comp_all)
repub_td <- tidy(repub_tdm)

# demo_total <- demo_td %>%
#   dplyr::group_by(term) %>%
#   dplyr::summarise(total = sum(count))
# 
# repub_total <- repub_td %>%
#   dplyr::group_by(term) %>%
#   dplyr::summarise(total = sum(count))

colnames(demo_td)=c("word","document","count")
demo_bing <- demo_td %>%
  left_join(get_sentiments("bing")) %>%
  filter(!is.na(sentiment))
```

```
## Joining, by = "word"
```

```r
colnames(repub_td)=c("word","document","count")
repub_bing <- repub_td %>%
  left_join(get_sentiments("bing")) %>%
  filter(!is.na(sentiment))
```

```
## Joining, by = "word"
```

```r
g_demo = ggplot(demo_bing, aes(x=factor(1), fill=sentiment))+
  geom_bar(width = 1)+
  coord_polar("y") + labs (x="",y="") + ggtitle("Democratic Senators' Tweets Responding to Barr") + theme(
plot.title = element_text(color="#0072B2", size=9.5, face="bold.italic"))

g_repub = ggplot(repub_bing, aes(x=factor(1), fill=sentiment))+
  geom_bar(width = 1)+
  coord_polar("y") + labs (x="",y="") + ggtitle("Republican Senators' Tweets Responding to Barr") + theme(
plot.title = element_text(color="#D55E00", size=9.5, face="bold.italic"))

library(ggpubr)
```

```
## Loading required package: magrittr
```

```
## 
## Attaching package: 'magrittr'
```

```
## The following object is masked from 'package:purrr':
## 
##     set_names
```

```
## The following object is masked from 'package:tidyr':
## 
##     extract
```

```r
ggarrange(g_demo, g_repub, 
           labels = c("A", "B"),
           ncol = 2, nrow = 1)
```

![](Datavis_4_files/figure-html/unnamed-chunk-8-1.png)<!-- -->

##3.Are you talking to me?

a) Identifying Re-Tweets


```r
st_retweet <- st[which(st$is_retweet=='TRUE'),]
st_retweetsub <- st_retweet[,c("screen_name","text","retweet_screen_name","retweet_text")]
name_and_party <- senator_DF[,c("Official_Twitter","Party_affiliation")]
colnames(name_and_party)<-c("retweet_screen_name","retweet_party")
# make subset12 which only contains retweets originating from other senators
subset11 <- dplyr::left_join(st_retweetsub,name_and_party,by="retweet_screen_name")
```

```
## Warning: Column `retweet_screen_name` joining character vector and factor,
## coercing into character vector
```

```r
subset12 <- drop_na(subset11)
```

Step1: Caculate by senator the amount of re-tweets they received and from which party these re-tweets came.


```r
count_retweet <- subset12 %>%
  dplyr::group_by(screen_name,retweet_party) %>%
  dplyr::summarise(count=n()) 
```

Step2: Visualize which parties the senators' retweets are from


```r
name_and_party_2 <- senator_DF[,c("Official_Twitter","Party_affiliation")]
colnames(name_and_party_2)<-c("screen_name","tweet_party")
subset13 <- dplyr::left_join(count_retweet,name_and_party_2,by="screen_name")
```

```
## Warning: Column `screen_name` joining character vector and factor, coercing
## into character vector
```

```r
subset14 <- drop_na(subset13)
demo_retweeted <- subset14[which(subset14$tweet_party=='Democratic Party'),]
repub_retweeted <- subset14[which(subset14$tweet_party=='Republican Party'),]
p_demo = ggplot(data=demo_retweeted,aes(screen_name,fill=retweet_party))
p_demo + geom_bar(aes(weight=count))+scale_fill_manual(values=c("#0072B2", "#009E73","#D55E00"))+ ggtitle("Count of Retweets for Democratic Senators")+labs(x="senator",
         y="retweet counts")+ coord_flip ()  + theme(
plot.title = element_text(color="black", size=14, face="bold.italic"),
axis.title.x = element_text(color="black", size=14, face="bold"),
axis.title.y = element_text(color="#0072B2", size=14, face="bold"),
axis.text.y = element_text(color="#0072B2",lineheight = 0.6
                                   , size = 5)
)
```

![](Datavis_4_files/figure-html/unnamed-chunk-11-1.png)<!-- -->


Comment 1: for senators in democratic party, most of them get retweets mostly from other senators in democratic party and only a small number of retweets from republican party senators(if any). However among them, SenBobCasey, SenatorTester, SenatorShaheen, Sen_JoeManchin, MarkWarner and amyklobuchar get re-tweeted on both sides of the aisle.


```r
p_repub = ggplot(data=repub_retweeted,aes(screen_name,fill=retweet_party))
p_repub + geom_bar(aes(weight=count))+scale_fill_manual(values=c("#0072B2", "#009E73","#D55E00"))+ ggtitle("Count of Retweets for Republican Senators")+labs(x="senator",
         y="retweet counts")+ coord_flip ()  + theme(
plot.title = element_text(color="black", size=14, face="bold.italic"),
axis.title.x = element_text(color="black", size=14, face="bold"),
axis.title.y = element_text(color="#D55E00", size=14, face="bold"),
axis.text.y = element_text(color="#D55E00",lineheight = 0.6
                                   , size = 5)
)
```

![](Datavis_4_files/figure-html/unnamed-chunk-12-1.png)<!-- -->

Comment 2: for senators in republican party, most of them get retweets mostly from other senators in republican party and only a small number of retweets from democratic party senators(if any). However among them, SenJohnKennedy, SenJohnHoeven,Senatorlsakson, SenatorCollins and RandPaul get re-tweeted on both sides of the aisle.

b) Identifying Mentions


```r
st_mention <- st_original[,c("screen_name","mentions_screen_name")]
st_mention_value <- st_mention[!is.na(st_mention$mentions_screen_name), ]
name_and_party_3 <- senator_DF[,c("Official_Twitter","Party_affiliation")]
colnames(name_and_party_3)<-c("mention_name","Party_affiliation")

# st_mention_value_2 <- st_mention_value %>% 
#     mutate(mentions_screen_name = strsplit(as.character(mentions_screen_name), ",")) %>% 
#     unnest(mentions_screen_name)
# st_mention_value_2$ mentions_screen_name <- gsub('[^a-zA-Z0-9.]', '', st_mention_value_2$ mentions_screen_name)

st_mention_value$mention_name <- gsub( '"', ',', st_mention_value$mentions_screen_name)
st_mention_value_4 <- st_mention_value[,c("screen_name","mention_name")]
st_mention_value_5 <- st_mention_value_4 %>% 
    mutate(mention_name = strsplit(as.character(mention_name), ",")) %>% 
    unnest(mention_name)
subset16 <- dplyr::left_join(st_mention_value_5,name_and_party_3,by="mention_name")
```

```
## Warning: Column `mention_name` joining character vector and factor,
## coercing into character vector
```

```r
subset17 <- drop_na(subset16)

# st_mention_value_3 <- st_mention_value %>% 
#     mutate(mentions_screen_name = strsplit(as.character(mentions_screen_name), ",")) %>% 
#     unnest(mentions_screen_name)

subset18 <- subset17[,c("screen_name","mention_name")]
network_mention <- graph_from_data_frame(subset18, directed=FALSE)
V(network_mention)$size <- centralization.degree(network_mention)$res
E(network_mention)$weight <- 1
g_mention <- igraph::simplify(network_mention, edge.attr.comb="sum")
library(intergraph)
set.seed(2103)
plotdata3 <- ggnetwork(g_mention, layout="fruchtermanreingold", 
          arrow.gap=0, cell.jitter=0)
# merge it with party information
plotdata4 <- merge(x = plotdata3, y = senatordata, by.x = "vertex.names", by.y = "Official_Twitter", all = TRUE)
library(ggplot2)
library(ggrepel)
plotdata4 <- drop_na(plotdata4)
(mention_g1 <- ggplot() +
  geom_edges(data=plotdata4, 
             aes(x=x, y=y, xend=xend, yend=yend,size= sqrt(weight) ),
             color="black", curvature=0.1,alpha=1/2) +
  geom_nodes(data=plotdata4,
             aes(x=x, y=y, xend=xend, yend=yend,size = sqrt(size),color = Party_affiliation)) +   
  geom_label_repel(data=unique(plotdata4[plotdata4$size>300,c(1,2,3)]),
                   aes(x=x, y=y, label=vertex.names), 
                   size=2, color="#8856a7") +
  theme_blank() +
  theme(legend.position="none") + scale_colour_manual(values=c("#0072B2", "#009E73","#D55E00")))
```

```
## Warning: Ignoring unknown aesthetics: xend, yend
```

![](Datavis_4_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

Comment: Senators from the same party are more likely to mention each other.
