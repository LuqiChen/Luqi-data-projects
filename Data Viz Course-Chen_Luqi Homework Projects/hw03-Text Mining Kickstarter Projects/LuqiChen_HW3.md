---
title: "Homework 3"
author: "luqi Chen"
date: "4/3/2019"
output:
  html_document:
    keep_md: true
---



##1.Identifying Successful Projects

```r
kickdata = read.csv("kickstarter_projects.csv")
# filter by top_category and identify success of project by looking at average number of backers
library(dplyr)
kickdata_category = kickdata %>%
  group_by(top_category) %>%
summarise(avg_backernumber=mean(backers_count)) 
kickdata_category
```

```
## # A tibble: 15 x 2
##    top_category avg_backernumber
##    <fct>                   <dbl>
##  1 art                      53.6
##  2 comics                  187. 
##  3 crafts                   44.2
##  4 dance                    54.8
##  5 design                  307. 
##  6 fashion                 125. 
##  7 film & video            114. 
##  8 food                     60.0
##  9 games                   494. 
## 10 journalism               53.8
## 11 music                    73.0
## 12 photography              51.4
## 13 publishing              114. 
## 14 technology              292. 
## 15 theater                  53.0
```

```r
library(ggplot2)
g1 = ggplot(data=kickdata_category,aes(x=reorder(top_category,avg_backernumber),y=avg_backernumber))
g1 + geom_bar(stat = "identity",fill ="darkgreen")+coord_flip()+ggtitle("Average backer number by category")+labs(x="Top Category", y="Average backer number") +theme(legend.position = "top")
```

![](Datavis_3_files/figure-html/unnamed-chunk-1-1.png)<!-- -->
As we can see from the above graph, in terms of number of backers, category "games","design" and "technology" were most successful in attracting funding.

##2.Writing your success story

a) Cleaning the Text and Word Cloud

```r
#install.packages("tm")
library(tm)
 #Load wordclou-d packagelibrary(tm)       # Framework for text mining.
#install.packages("quanteda")
library(quanteda) # Another great text mining package
#install.packages("tidytext")
library(tidytext) # Text as Tidy Data (good for use with ggplot2)
# select 1000 most successful projects and 1000 unsuccessful projects
successdata = kickdata[which(kickdata$state=='successful'),]
success_ranked <- successdata %>%
arrange(desc(.$backers_count))
success_1000 = success_ranked[1:1000,] #define the "successful" projects with the most backers as the most unsucessful ones
unsuccessdata = kickdata[which(kickdata$state=='failed'),]
unsuccess_ranked <- unsuccessdata %>%
arrange(.$backers_count)
unsuccess_1000 = unsuccess_ranked[1:1000,] #define the "failed" projects with the least backers as the most unsucessful ones
# merge the 1000 most successful and 1000 most unsuccessful projects together
data2000 = rbind(success_1000, unsuccess_1000)
# extract "blurb"
data2000_blurb = data2000[,"blurb"]
# add doc_id
IDdf = as.data.frame(c(1:2000))
text2000 = cbind(IDdf,data2000_blurb)
library(plyr)
```

```
## -------------------------------------------------------------------------
```

```
## You have loaded plyr after dplyr - this is likely to cause problems.
## If you need functions from both plyr and dplyr, please load plyr first, then dplyr:
## library(plyr); library(dplyr)
```

```
## -------------------------------------------------------------------------
```

```
## 
## Attaching package: 'plyr'
```

```
## The following objects are masked from 'package:dplyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
```

```
## The following object is masked from 'package:purrr':
## 
##     compact
```

```r
colnames(text2000) = c("doc_id", "text")
df2000_source = DataframeSource(text2000)
# Convert df_source to a corpus: df2000_corpus
df2000_corpus <- VCorpus(df2000_source)
# Define clean_corpus function
removeNumPunct <- function(x){gsub("[^[:alpha:][:space:]]*", "", x)}
clean_corpus <- function(corpus){
  corpus <- tm_map(corpus, content_transformer(tolower))
  corpus <- tm_map(corpus, removeWords, c(stopwords("en")))  
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, content_transformer(removeNumPunct))
  corpus <- tm_map(corpus, stripWhitespace)
  return(corpus)
}
# Apply the function to df2000_corpus
df2000_clean <- clean_corpus(df2000_corpus)
# Stem the words left
library(SnowballC)
text2000_stemmed = tm_map(df2000_clean, stemDocument)
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
text2000_comp_all <- mclapply(text2000_stemmed, 
                          stemCompletion2, 
                          dictionary = df2000_clean, 
                          mc.cores=no_cores)
text2000_comp_all <- as.VCorpus(text2000_comp_all)
text2000_dtm <- DocumentTermMatrix(text2000_comp_all)
text2000_dtm
```

```
## <<DocumentTermMatrix (documents: 2000, terms: 5226)>>
## Non-/sparse entries: 21564/10430436
## Sparsity           : 100%
## Maximal term length: 61
## Weighting          : term frequency (tf)
```

```r
# Convert text2000_dtm to a matrix: text2000_m
text2000_m = as.matrix(text2000_dtm)
dim(text2000_m)
```

```
## [1] 2000 5226
```

```r
text2000_m[1:3,100:103]
```

```
##               Terms
## Docs           alert alex alexa alexanoise
##   character(0)     0    0     0          0
##   character(0)     0    0     0          0
##   character(0)     0    0     0          0
```

```r
# Make a TDM for successful projects
# extract "blurb"
data1000_blurb = success_1000[,"blurb"]
# add doc_id
IDdf2 = as.data.frame(c(1:1000))
text1000 = cbind(IDdf2,data1000_blurb)
library(plyr)
colnames(text1000) = c("doc_id", "text")
df1000_source = DataframeSource(text1000)
# Convert df_source to a corpus: df1000_corpus
df1000_corpus <- VCorpus(df1000_source)
# Apply the clean function to df1000_corpus
df1000_clean <- clean_corpus(df1000_corpus)
# Stem the words left
text1000_stemmed = tm_map(df1000_clean, stemDocument)
# Complete the stems, use stemCompletion2 again
text1000_comp_all <- mclapply(text1000_stemmed, 
                          stemCompletion2, 
                          dictionary = df2000_clean, 
                          mc.cores=no_cores)
text1000_comp_all <- as.VCorpus(text1000_comp_all)
text1000_tdm = TermDocumentMatrix(text1000_comp_all)
text1000_td = tidy(text1000_tdm)
library(dplyr)
text1000_td_df2 = as_data_frame(text1000_td)
```

```
## Warning: `as_data_frame()` is deprecated, use `as_tibble()` (but mind the new semantics).
## This warning is displayed once per session.
```

```r
library(tidytext)
text1000_n <- text1000_td_df2 %>%
  dplyr::group_by(term) %>%
  dplyr::summarise(total = sum(count)) %>%
  top_n(50) 
```

```
## Selecting by total
```

```r
library(wordcloud)
# Set seed - to make your word cloud reproducible 
set.seed(2103)
# Create a wordcloud 
wordcloud(text1000_n$term, text1000_n$total, 
         max.words = 30, colors = "red")
```

![](Datavis_3_files/figure-html/unnamed-chunk-2-1.png)<!-- -->

b) Success in words

```r
# get the dataframe of the turn frequency of 1000 most successful projects
text1000_n1 = text1000_td %>%
  dplyr::group_by(term) %>%
  dplyr::summarise(n1 = sum(count)) 
# get the datafraome of the turn frequency of 1000 most unsuccessful projects
# extract "blurb"
data10002_blurb = unsuccess_1000[,"blurb"]
# add doc_id
text10002 = cbind(IDdf2,data10002_blurb)
library(plyr)
colnames(text10002) = c("doc_id", "text")
df10002_source = DataframeSource(text10002)
# Convert df_source to a corpus: df1000_corpus
df10002_corpus <- VCorpus(df10002_source)
# Apply the clean function to df10002_corpus
df10002_clean <- clean_corpus(df10002_corpus)
# Stem the words left
text10002_stemmed = tm_map(df10002_clean, stemDocument)
# Complete the stems, use stemCompletion2 again
library(parallel)
library(pbapply)
text10002_comp_all <- mclapply(text10002_stemmed, 
                          stemCompletion2, 
                          dictionary = df2000_clean, 
                          mc.cores=no_cores)
text10002_comp_all <- as.VCorpus(text10002_comp_all)
text10002_tdm = TermDocumentMatrix(text10002_comp_all)
text10002_td = tidy(text10002_tdm)
library(dplyr)
text10002_td_df = as.data.frame(text10002_td)
library(tidytext)
text1000_n2 = text10002_td_df %>%
  dplyr::group_by(term) %>%
  dplyr::summarise(n2 = sum(count))
textfull = left_join(text1000_n1,text1000_n2,by="term")
textjoin = na.omit(textfull)
textjoin$n12 = textjoin$n1 + textjoin$n2
textjoin_arranged = textjoin %>%
 arrange(desc(.$n12))
textjoin_top15 = textjoin_arranged[1:15,]
library(plotrix)
p_2b <- pyramid.plot(textjoin_top15$n1, textjoin_top15$n2, labels = textjoin_top15$term, 
             gap = 10, top.labels = c("successful projects", " ", "unsuccessful projects"), 
             main = "Words in Common", laxlab = NULL, 
             raxlab = NULL, unit = NULL, labelcex=0.5)
```

![](Datavis_3_files/figure-html/unnamed-chunk-3-1.png)<!-- -->

c) Simplicity as a virtue

```r
# Use the corpus before cleaning to do the analysis
packages <- c("devtools","knitr","tidyverse","widgetframe","readr",
              "wordcloud", "base64enc", "tm", "quanteda", 
              "qdapDictionaries", "tidytext", 
              "stats","manifestoR","readtext",
              "rvest", "stringr", 
              "SnowballC", "plotrix", "tidyr", "tidytext", "stats", 
              "dendextend", "ggthemes",
              "httr","jsonlite")

 packages <- lapply(packages, FUN = function(x) {
  if(!require(x, character.only = TRUE)) {
    install.packages(x)
  library(x, character.only = TRUE)
  }
}
)
```

```
## Loading required package: manifestoR
```

```
## When publishing work using the Manifesto Corpus, please make sure to cite it correctly and to give the identification number of the corpus version used for your analysis.
## 
## You can print citation and version information with the function mp_cite().
## 
## Note that some of the scaling/analysis algorithms provided with this package were conceptually developed by authors referenced in the respective function documentation. Please also reference them when using these algorithms.
```

```
## 
## Attaching package: 'manifestoR'
```

```
## The following object is masked from 'package:plotrix':
## 
##     rescale
```

```
## Loading required package: readtext
```

```
## Loading required package: httr
```

```
## 
## Attaching package: 'httr'
```

```
## The following object is masked from 'package:NLP':
## 
##     content
```

```
## Loading required package: jsonlite
```

```
## 
## Attaching package: 'jsonlite'
```

```
## The following object is masked from 'package:purrr':
## 
##     flatten
```

```r
require(quanteda)
require(dplyr)
corpus3c <- corpus(df2000_corpus)
FRE_corpus3c <- textstat_readability(corpus3c,
              measure=c('Flesch.Kincaid'))
MYFRE <- data_frame(FK = FRE_corpus3c$Flesch.Kincaid,
    backer_number = data2000$backers_count)
```

```
## Warning: `data_frame()` is deprecated, use `tibble()`.
## This warning is displayed once per session.
```

```r
ggplot(data=MYFRE, aes(x=backer_number,y=FK)) + 
  geom_point(alpha=0.5) + geom_smooth() + guides(size=FALSE) +
  theme_tufte(22) + xlab("backer number") + ylab("Flesch-Kincaid Grade Level") + theme(legend.position="none")
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
```

![](Datavis_3_files/figure-html/unnamed-chunk-4-1.png)<!-- -->
Finding: it seems like there is not a strong relationship between backer number and Flesch-Kincaid Grade Level

##3.Sentiment
a) Stay positive

```r
pos <- read.table("data/dictionaries/positive-words.txt", as.is=T)
neg <- read.table("data/dictionaries/negative-words.txt", as.is=T)

# meta(df2000_corpus) <- data2000
# # Add data frame to Corpus
# # Index of the metadata for a document in the corpus in   
# meta(df2000_corpus, type="local", tag="id") <- data2000$id
# meta(df2000_corpus, type="local", tag="backer") <- data2000$backers_count
# 
# # Re-attach metadata to Corpus
# data2000[] <- lapply(data2000, as.character)
# for (i in 1:dim(data2000)[1]){
#   text2000_comp_all[[i]]$meta$ID <- data2000[i,"id"]
#   text2000_comp_all[[i]]$meta$BACKER <- data2000[i,"backers_count"]
# }
# 
# text2000_comp_all_ID <- as.VCorpus(text2000_comp_all)
# text2000_ID_tdm <- TermDocumentMatrix(text2000_comp_all_ID)
# text2000_ID_td <- tidy(text2000_ID_tdm)

# # Get meta data
# meta <- as_data_frame(str_split_fixed(text2000_ID_td$document, "_", n=3))
# colnames(meta) <- c("president", "year", "party")
# # Merge on
# sotu_td <- as_data_frame(cbind(sotu_td, meta))
# 
# doc_id = c(rep("success",1000),rep("fail",1000))
# DF3a <- data.frame(doc_id, text = data2000$blurb, stringsAsFactors = FALSE)
# DF3a_source = DataframeSource(DF3a)
# # Convert df_source to a corpus
# DF3a_corpus <- VCorpus(DF3a_source)
# DF3a_clean <- clean_corpus(DF3a_corpus)
# library(SnowballC)
# DF3a_stemmed = tm_map(DF3a_clean, stemDocument)
# DF3a_comp_all <- mclapply(DF3a_stemmed, 
#                           stemCompletion2, 
#                           dictionary = df2000_clean, 
#                           mc.cores=no_cores)
# DF3a_comp_all <- as.VCorpus(DF3a_comp_all)
# DF3a_tdm = TermDocumentMatrix(DF3a_comp_all)
# DF3a_td = tidy(DF3a_tdm)

tok <- quanteda::tokens(corpus3c[1:2000])
tone = numeric()
for (i in 1:2000){
  pos.count <- sum(tok[[i]]%in%pos[,1])
  neg.count <- sum(tok[[i]]%in%neg[,1])
  out <- (pos.count - neg.count)/(pos.count+neg.count)
  tone <- c(tone, out)
  tone <- replace(tone, is.na(tone),0)
}
tone_DF = data.frame(tone)
dataframe3a = cbind(tone_DF,data2000)
ggplot(data=dataframe3a, aes(x=backers_count,y=tone)) + 
  geom_point(alpha=0.5) + geom_smooth() + guides(size=FALSE) + theme_tufte(22) + xlab("backer number") + ylab("Tone Score") + theme(legend.position="none")
```

```
## `geom_smooth()` using method = 'gam' and formula 'y ~ s(x, bs = "cs")'
```

![](Datavis_3_files/figure-html/unnamed-chunk-5-1.png)<!-- -->
It seems like there is not strong relationship between tone of the document and success.

b) Positive VS negative

```r
positiveDF <- dataframe3a[which(dataframe3a$tone>0),]
negativeDF <- dataframe3a[which(dataframe3a$tone<0),]
positiveDF_blurb <- positiveDF[,"blurb"]
positivetext <- paste(positiveDF_blurb, collapse =" ")
negativeDF_blurb <- negativeDF[,"blurb"]
negativetext <- paste(negativeDF_blurb, collapse =" ")
id3b = c(1,2)
alltext = c(positivetext,negativetext)
dataframe3b = data.frame(id3b, alltext)
colnames(dataframe3b) = c("doc_id", "text")
dataframe3b_source = DataframeSource(dataframe3b)
dataframe3b_corpus <- VCorpus(dataframe3b_source)
dataframe3b_clean <- clean_corpus(dataframe3b_corpus)
library(SnowballC)
dataframe3b_stemmed = tm_map(dataframe3b_clean, stemDocument)
library(pbapply) 
# Calculate the number of cores
no_cores <- detectCores() - 1
# Initiate cluster
dataframe3b_comp_all <- mclapply(dataframe3b_stemmed, 
                              stemCompletion2, 
                              dictionary = dataframe3b_clean, 
                              mc.cores=no_cores)
dataframe3b_comp_all <- as.VCorpus(dataframe3b_comp_all)
dataframe3b_dtm <- DocumentTermMatrix(dataframe3b_comp_all)
dataframe3b_m = as.matrix(dataframe3b_dtm)
dim(dataframe3b_m)
```

```
## [1]    2 3707
```

```r
dataframe3b_tdm <- TermDocumentMatrix(dataframe3b_comp_all)
dataframe3b_m2 = as.matrix(dataframe3b_tdm)
colnames(dataframe3b_m2) = c("positive", "negative")
comparison.cloud(dataframe3b_m2, colors = c("orange", "blue"), 
                 scale=c(0.1,2), title.size= 1, 
                 max.words = 100)
```

```
## Warning in comparison.cloud(dataframe3b_m2, colors = c("orange", "blue"), :
## anthology could not be fit on page. It will not be plotted.
```

![](Datavis_3_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

c) Get in their mind

```r
text1000_n1$status = c("success")
text1000_n2$status = c("unsuccess")
text1000_1_3c = text1000_n1[,c("status","term")]
text1000_2_3c = text1000_n2[,c("status","term")]
data3c = rbind(text1000_1_3c, text1000_2_3c)
colnames(data3c) = c("status", "word")

require(tidytext)
NRC_DF <- data3c %>%
  left_join(get_sentiments("nrc")) %>%
  filter(!is.na(sentiment)) 
```

```
## Joining, by = "word"
```

```r
NRC_DF_group = NRC_DF %>%
  dplyr::group_by(sentiment,status) %>%
  dplyr::summarize(wordnumber = n())

g = ggplot(data=NRC_DF_group,aes(sentiment,fill=status))
g + geom_bar(aes(weight=wordnumber),position = "dodge")+scale_fill_manual(values=c("darkgreen", "orange"))+ggtitle("words and success")+labs(x="sentiment",
         y="Word Number") 
```

![](Datavis_3_files/figure-html/unnamed-chunk-7-1.png)<!-- -->
Finding: It seems like there is not strong relationship between sentiment category and "success or not", for each categroy, the word count for success projects and unsuccess projects is only slightly different.
