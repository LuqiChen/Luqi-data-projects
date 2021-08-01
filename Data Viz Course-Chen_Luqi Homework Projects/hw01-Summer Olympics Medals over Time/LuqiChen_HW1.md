---
title: "Homework 1 - Summer Olympics"
author: "luqi Chen"
date: "2/21/2019"
output:
  html_document:
    keep_md: true
---


## 1.Medal Counts over Time

```r
setwd("/Users/luqi/Desktop")
ath = read.csv("athletes_and_events.csv")
nocregions = read.csv("noc_regions.csv")
gdppop = read.csv("gdp_pop.csv")
#install.packages("tidyverse")
```

```r
athsummer <- ath[which(ath$Season=='Summer'),]
library(tidyr)
tathsummer = drop_na(athsummer, Medal)
tathsummer= tathsummer[!duplicated(tathsummer), ]
```



```r
library(dplyr)
```

```
## Warning: package 'dplyr' was built under R version 3.5.2
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
#group_by(athsummer,NOC, add = FALSE)
cleandata <- tathsummer %>%
  group_by(NOC) %>%
summarize(medal_number = length(Medal),gold_number = length(which(Medal=="Gold")),silver_number = length(which(Medal=="Silver")),bronze_number = length(which(Medal=="Bronze"))) %>%
  arrange(desc(.$medal_number))
#cleandata
```

```r
tathsummerUSA <- ath[which(tathsummer$NOC=='USA'),]
cleandataUSA <- tathsummerUSA %>%
  group_by(Year,Sex) %>%
summarize(gold_number = length(which(Medal=="Gold")),silver_number = length(which(Medal=="Silver")),bronze_number = length(which(Medal=="Bronze"))) 
cleandataUSA
```

```
## # A tibble: 65 x 5
## # Groups:   Year [35]
##     Year Sex   gold_number silver_number bronze_number
##    <int> <fct>       <int>         <int>         <int>
##  1  1896 M               0             0             0
##  2  1900 M               4             8             2
##  3  1904 M               1             3             1
##  4  1906 M               1             1             3
##  5  1908 F               0             0             0
##  6  1908 M               5             3             4
##  7  1912 M              10             8            10
##  8  1920 F               0             0             1
##  9  1920 M               6             8             8
## 10  1924 F               0             0             0
## # … with 55 more rows
```


```r
library(ggplot2)
# ggplot(data=cleandataUSA,
#             aes(x=Year,
#                 y=gold_number))+
#   geom_line(aes(group=Sex,
#                   color=Sex))
# look at USA's gold medal number by time trend, considering the gneder of the gold medal winners
p = ggplot(data=cleandataUSA,
            aes(x=Year,
                y=gold_number))
p + geom_line(aes(group=Sex,
                  color=Sex))+
         labs(x="Year",
         y="Gold Medal Number",
         color="Gender") +
   ggtitle("USA's Gold Medal Number by Sex")+ geom_point(size=2,aes(color=Sex))
```

![](Homework1_files/figure-html/unnamed-chunk-5-1.png)<!-- -->

```r
cleandata_add_Medal <- tathsummer %>%
  group_by(NOC,Medal) %>%
summarize(medal_number = length(Medal))
#cleandata_add_Medal
cleandata_add_Medal_top5 <- cleandata_add_Medal[which(cleandata_add_Medal$NOC == 'USA'|cleandata_add_Medal$NOC =='URS'|cleandata_add_Medal$NOC =='GBR'|cleandata_add_Medal$NOC =='GER'|cleandata_add_Medal$NOC =='FRA'),]
cleandata_add_Medal_top5
```

```
## # A tibble: 15 x 3
## # Groups:   NOC [5]
##    NOC   Medal  medal_number
##    <fct> <fct>         <int>
##  1 FRA   Bronze          587
##  2 FRA   Gold            463
##  3 FRA   Silver          567
##  4 GBR   Bronze          620
##  5 GBR   Gold            635
##  6 GBR   Silver          729
##  7 GER   Bronze          649
##  8 GER   Gold            592
##  9 GER   Silver          538
## 10 URS   Bronze          596
## 11 URS   Gold            832
## 12 URS   Silver          635
## 13 USA   Bronze         1197
## 14 USA   Gold           2472
## 15 USA   Silver         1333
```

```r
g = ggplot(data=cleandata_add_Medal_top5,aes(NOC,fill=Medal))
g + geom_bar(aes(weight=medal_number),position = "dodge")+scale_fill_manual(values=c("darkred", "gold","gray40"))+ggtitle("Top 5 Medal winning NOCs")+labs(x="NOC",
         y="Medal Number") 
```

![](Homework1_files/figure-html/unnamed-chunk-6-1.png)<!-- -->

```r
# I would recommend the first visualization because we can see some interesting pattern in the graph, as time goes by, in USA, the gap of medal numbers won by male and female is narrowing. 
```
##2.Medal Counts adjusted by Population, GDP

```r
tath = drop_na(ath, Medal)
tath = tath[!duplicated(tath), ]
gdppop=gdppop[!duplicated(gdppop), ]
addvalue<- tath %>%
  group_by(NOC) %>%
summarize(medal_number = length(Medal),gold_number = length(which(Medal=="Gold")),silver_number = length(which(Medal=="Silver")),bronze_number = length(which(Medal=="Bronze")),medal_value = 3*as.numeric(gold_number) + 2*as.numeric(silver_number)+as.numeric(bronze_number)) %>%
  arrange(desc(.$medal_number))
#addvalue

fulltable= merge(x = addvalue, y = gdppop, by.x = "NOC", by.y = "Code", all = TRUE)
#fulltable
fulltable$medal_value_byGDP = (as.numeric(fulltable$medal_value)/as.numeric(fulltable$GDP.per.Capita))
fulltable$medal_value_byPOP = (as.numeric(fulltable$medal_value)/as.numeric(fulltable$Population))*1000000
fulltable$highlight <- fulltable$NOC == 'CHN'
fulltable <- fulltable %>% 
  mutate(highlight=replace(highlight, highlight == "TRUE", "CHN"), highlight=replace(highlight, highlight == "FALSE", "OTHER"))
Ranking_plot1=ggplot(fulltable, aes(x = GDP.per.Capita, y = medal_value)) + 
  geom_point(aes(colour = highlight)) + 
  scale_colour_manual(values = c("OTHER" = "black", "CHN" = "red"))+ 
  labs(x="GDP.per.Capita",
         y="Unadjusted Medal Value",
         color="NOC") +
   ggtitle("CHN's Ranking in Unadjusted Medal Value")
Ranking_plot1
```

```
## Warning: Removed 99 rows containing missing values (geom_point).
```

![](Homework1_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

```r
Ranking_plot2=ggplot(fulltable, aes(x = GDP.per.Capita, y = medal_value_byGDP)) + 
  geom_point(aes(colour = highlight)) + 
  scale_colour_manual(values = c("OTHER" = "black", "CHN" = "red"))+ 
  labs(x="GDP.per.Capita",
         y="Medal Value Adjusted by GDP",
         color="NOC") +
   ggtitle("CHN's Ranking in Adjusted Medal Value(by GDP)")
Ranking_plot2
```

```
## Warning: Removed 99 rows containing missing values (geom_point).
```

![](Homework1_files/figure-html/unnamed-chunk-7-2.png)<!-- -->

```r
Ranking_plot3=ggplot(fulltable, aes(x = GDP.per.Capita, y = medal_value_byPOP)) + 
  geom_point(aes(colour = highlight)) + 
  scale_colour_manual(values = c("OTHER" = "black", "CHN" = "red"))+
  labs(x="GDP.per.Capita",
         y="Medal Value Adjusted by Population",
         color="NOC") +
   ggtitle("CHN's Ranking in Adjusted Medal Value(by Population)")
Ranking_plot3
```

```
## Warning: Removed 99 rows containing missing values (geom_point).
```

![](Homework1_files/figure-html/unnamed-chunk-7-3.png)<!-- -->

```r
library(ggplot2)
library(gridExtra)
```

```
## 
## Attaching package: 'gridExtra'
```

```
## The following object is masked from 'package:dplyr':
## 
##     combine
```

```r
grid.arrange(Ranking_plot1, Ranking_plot2, Ranking_plot3,nrow=3)
```

```
## Warning: Removed 99 rows containing missing values (geom_point).

## Warning: Removed 99 rows containing missing values (geom_point).

## Warning: Removed 99 rows containing missing values (geom_point).
```

![](Homework1_files/figure-html/unnamed-chunk-7-4.png)<!-- -->

```r
## From the comparison of 3 plots we can see that CHN ranks top when medal_value is adjusted by GDP, and ranks very low when medal_value is adjusted by population, which shows that CHN has a really large population.
```
##3.Host Country Advantage

```r
library(rvest)
```

```
## Loading required package: xml2
```

```r
library(stringr)
wiki_hosts <- read_html("https://en.wikipedia.org/wiki/Summer_Olympic_Games") 
hosts <- html_table(html_nodes(wiki_hosts, "table")[[8]], fill=TRUE)
hosts <- hosts[-1,1:3]
hosts$city <- str_split_fixed(hosts$Host, n=2, ",")[,1]
hosts$country <- str_split_fixed(hosts$Host, n=2, ", ")[,2]
tathsummer_host <- tathsummer[which(tathsummer$NOC =='FRA'|tathsummer$NOC =='USA'|tathsummer$NOC == 'GBR'|tathsummer$NOC =='SWE'|tathsummer$NOC =='BEL'|tathsummer$NOC =='NED'|tathsummer$NOC =='GER'|tathsummer$NOC =='GBR'|tathsummer$NOC =='FIN'|tathsummer$NOC =='ANZ'|tathsummer$NOC =='ITA'|tathsummer$NOC =='JPN'|tathsummer$NOC =='MEX'|tathsummer$NOC =='CAN'|tathsummer$NOC =='URS'|tathsummer$NOC =='KOR'|tathsummer$NOC =='ESP'|tathsummer$NOC =='GRE'|tathsummer$NOC =='CHN'|tathsummer$NOC =='BRA'),]
#tathsummer_host
tathsummer_host_addyear<- tathsummer_host %>%
  group_by(NOC,Year) %>%
summarize(medal_number = length(Medal))
#tathsummer_host_addyear
tathsummer_host_group<- tathsummer_host %>%
  group_by(NOC) %>%
summarize(medal_number = length(Medal))
#tathsummer_host_group
NOC = c("ANZ","ANZ","BEL","BEL","BRA","BRA","CAN","CAN","CHN","CHN","ESP","ESP","FIN","FIN","FRA","FRA","GBR","BGR","GER","GER","GRE","GRE","ITA","ITA","JPN","JPN","KOR","KOR","MEX","MEX","NED","NED","SWE","SWE","URS","URS","USA","USA")
Host = c("true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false","true","false")
Yearly_Average_MedalNumber = c(0,1.1,188,9.9,50,15.7,23,26.6,184,26.9,69,15.5,40,16.1,172.5,49.3,247,57.3,224,91.4,39.5,5.6,88,50.3,62,29.2,77,17.6,9,3.7,57,31.9,190,34,442,60,298.5,158.7)
df = data.frame(NOC,Host,Yearly_Average_MedalNumber)
df
```

```
##    NOC  Host Yearly_Average_MedalNumber
## 1  ANZ  true                        0.0
## 2  ANZ false                        1.1
## 3  BEL  true                      188.0
## 4  BEL false                        9.9
## 5  BRA  true                       50.0
## 6  BRA false                       15.7
## 7  CAN  true                       23.0
## 8  CAN false                       26.6
## 9  CHN  true                      184.0
## 10 CHN false                       26.9
## 11 ESP  true                       69.0
## 12 ESP false                       15.5
## 13 FIN  true                       40.0
## 14 FIN false                       16.1
## 15 FRA  true                      172.5
## 16 FRA false                       49.3
## 17 GBR  true                      247.0
## 18 BGR false                       57.3
## 19 GER  true                      224.0
## 20 GER false                       91.4
## 21 GRE  true                       39.5
## 22 GRE false                        5.6
## 23 ITA  true                       88.0
## 24 ITA false                       50.3
## 25 JPN  true                       62.0
## 26 JPN false                       29.2
## 27 KOR  true                       77.0
## 28 KOR false                       17.6
## 29 MEX  true                        9.0
## 30 MEX false                        3.7
## 31 NED  true                       57.0
## 32 NED false                       31.9
## 33 SWE  true                      190.0
## 34 SWE false                       34.0
## 35 URS  true                      442.0
## 36 URS false                       60.0
## 37 USA  true                      298.5
## 38 USA false                      158.7
```

```r
Adv_plot=ggplot(df, aes(x = NOC, y = Yearly_Average_MedalNumber )) + 
  geom_point(aes(colour = Host)) + 
   ggtitle("Host Country Advantage")
Adv_plot
```

![](Homework1_files/figure-html/unnamed-chunk-8-1.png)<!-- -->


##4. Most successful athletes

```r
namedata <- tath %>%
  group_by(Name,Medal,Sex) %>%
summarize(medal_number = length(Medal),gold_number = length(which(Medal=="Gold")),silver_number = length(which(Medal=="Silver")),bronze_number = length(which(Medal=="Bronze"))) %>%
  arrange(desc(.$medal_number))
#namedata
## I define "most successful athletes"as athletes who won most gold medals, let's look at top 10 gold medal winners and see the gender distribution
namedata_top10 = namedata[1:10,]
namedata_top10
```

```
## # A tibble: 10 x 7
## # Groups:   Name, Medal [10]
##    Name    Medal Sex   medal_number gold_number silver_number bronze_number
##    <fct>   <fct> <fct>        <int>       <int>         <int>         <int>
##  1 Michae… Gold  M               23          23             0             0
##  2 "Raymo… Gold  M               10          10             0             0
##  3 "Frede… Gold  M                9           9             0             0
##  4 Larysa… Gold  F                9           9             0             0
##  5 Mark A… Gold  M                9           9             0             0
##  6 Paavo … Gold  M                9           9             0             0
##  7 Birgit… Gold  F                8           8             0             0
##  8 "Jenni… Gold  F                8           8             0             0
##  9 "Matth… Gold  M                8           8             0             0
## 10 Ole Ei… Gold  M                8           8             0             0
```

```r
g2 = ggplot(data=namedata_top10,aes(Name,fill=Sex))
g2 + geom_bar(aes(weight=gold_number),position = position_stack(reverse = TRUE))+coord_flip()+ggtitle("Top 10 Gold-Medal winning Atheletes")+labs(x="Name",
         y="Gold Medal Number") +theme(legend.position = "top")
```

![](Homework1_files/figure-html/unnamed-chunk-9-1.png)<!-- -->

```r
namedata_addsport_year <- tath %>%
  group_by(Name,Sport,Year) %>%
summarize(medal_number = length(Medal),gold_number = length(which(Medal=="Gold"))) %>%
 arrange(desc(.$gold_number))
#namedata_addsport_year
namedata_addsport_year_top10 <- namedata_addsport_year[which(namedata_addsport_year$Name == 'Michael Fred Phelps, II'|namedata_addsport_year$Name == 'Raymond Clarence "Ray" Ewry'|namedata_addsport_year$Name == 'Frederick Carlton "Carl" Lewis'|namedata_addsport_year$Name == 'Larysa Semenivna Latynina (Diriy-)'|namedata_addsport_year$Name == 'Mark Andrew Spitz'|namedata_addsport_year$Name == 'Paavo Johannes Nurmi'|namedata_addsport_year$Name == 'Birgit Fischer-Schmidt'|namedata_addsport_year$Name == 'Jennifer Elisabeth "Jenny" Thompson (-Cumpelik)'|namedata_addsport_year$Name == 'Matthew Nicholas "Matt" Biondi'|namedata_addsport_year$Name == 'Ole Einar Bjrndalen'),]
p2 <- ggplot(data=namedata_addsport_year_top10,
            aes(x=Year,
                y=gold_number))
p2 + geom_line(aes(group=Name,
                  color=Sport)) +
    labs(x="Year",
         y="Gold Medal Number",
         color="Sport") + geom_point(size=4,aes(color=Sport)) + ggtitle("Top 10 Gold-Medal winning Atheletes' Sports")
```

![](Homework1_files/figure-html/unnamed-chunk-9-2.png)<!-- -->

```r
## For this graph, every tiny line is an athlete, my interesting finding is that before 1925, top athletes are crazy about winning medals in Atheletics, and later on, the popular sport became swimming
```
##5.Make two plots interactive

```r
library(devtools)
devtools::install_github("ropensci/plotly",force=TRUE)
```

```
## Downloading GitHub repo ropensci/plotly@master
## from URL https://api.github.com/repos/ropensci/plotly/zipball/master
```

```
## Installing plotly
```

```
## '/Library/Frameworks/R.framework/Resources/bin/R' --no-site-file  \
##   --no-environ --no-save --no-restore --quiet CMD INSTALL  \
##   '/private/var/folders/rx/w35t6lg902l19fdtv5wqjfgr0000gn/T/Rtmpqu1upQ/devtools7c0319d4b1f/ropensci-plotly-c05f001'  \
##   --library='/Library/Frameworks/R.framework/Versions/3.5/Resources/library'  \
##   --install-tests
```

```
## 
```

```r
R.home(component = "home")
```

```
## [1] "/Library/Frameworks/R.framework/Resources"
```

```r
#install.packages("usethis")
library(usethis)
```

```
## 
## Attaching package: 'usethis'
```

```
## The following objects are masked from 'package:devtools':
## 
##     use_appveyor, use_build_ignore, use_code_of_conduct,
##     use_coverage, use_cran_badge, use_cran_comments, use_data,
##     use_data_raw, use_dev_version, use_git, use_git_hook,
##     use_github, use_github_links, use_gpl3_license,
##     use_mit_license, use_news_md, use_package, use_package_doc,
##     use_rcpp, use_readme_md, use_readme_rmd, use_revdep,
##     use_rstudio, use_test, use_testthat, use_travis, use_vignette
```

```r
usethis::edit_r_environ()
```

```
## ● Edit /Users/luqi/.Renviron
## ● Restart R for changes to take effect
```

```r
Sys.setenv("plotly_username"="luqi.chen")
Sys.setenv("plotly_api_key"="R9LBpABPVy7aUJ3Sx7jf")
interation_1=p + geom_line(aes(group=Sex,
                  color=Sex))+
         labs(x="Year",
         y="Gold Medal Number",
         color="Gender") +
   ggtitle("USA's Gold Medal Number by Sex")+ geom_point(size=2,aes(color=Sex))
#install.packages('plotly')
library(plotly)
```

```
## 
## Attaching package: 'plotly'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     last_plot
```

```
## The following object is masked from 'package:stats':
## 
##     filter
```

```
## The following object is masked from 'package:graphics':
## 
##     layout
```

```r
ggplotly(interation_1)
```

<!--html_preserve--><div id="htmlwidget-bd2f77c2d97b9c1db36b" style="width:672px;height:480px;" class="plotly html-widget"></div>
<script type="application/json" data-for="htmlwidget-bd2f77c2d97b9c1db36b">{"x":{"data":[{"x":[1908,1920,1924,1928,1932,1936,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1994,1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,2016],"y":[0,0,0,0,0,0,0,0,0,2,2,0,0,2,2,9,3,2,0,3,1,5,5,4,3,3,4,4,2,4],"text":["Sex: F<br />Sex: F<br />Year: 1908<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1920<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1924<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1928<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1932<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1936<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1948<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1952<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1956<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1960<br />gold_number:  2","Sex: F<br />Sex: F<br />Year: 1964<br />gold_number:  2","Sex: F<br />Sex: F<br />Year: 1968<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1972<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1976<br />gold_number:  2","Sex: F<br />Sex: F<br />Year: 1980<br />gold_number:  2","Sex: F<br />Sex: F<br />Year: 1984<br />gold_number:  9","Sex: F<br />Sex: F<br />Year: 1988<br />gold_number:  3","Sex: F<br />Sex: F<br />Year: 1992<br />gold_number:  2","Sex: F<br />Sex: F<br />Year: 1994<br />gold_number:  0","Sex: F<br />Sex: F<br />Year: 1996<br />gold_number:  3","Sex: F<br />Sex: F<br />Year: 1998<br />gold_number:  1","Sex: F<br />Sex: F<br />Year: 2000<br />gold_number:  5","Sex: F<br />Sex: F<br />Year: 2002<br />gold_number:  5","Sex: F<br />Sex: F<br />Year: 2004<br />gold_number:  4","Sex: F<br />Sex: F<br />Year: 2006<br />gold_number:  3","Sex: F<br />Sex: F<br />Year: 2008<br />gold_number:  3","Sex: F<br />Sex: F<br />Year: 2010<br />gold_number:  4","Sex: F<br />Sex: F<br />Year: 2012<br />gold_number:  4","Sex: F<br />Sex: F<br />Year: 2014<br />gold_number:  2","Sex: F<br />Sex: F<br />Year: 2016<br />gold_number:  4"],"type":"scatter","mode":"lines","line":{"width":1.88976377952756,"color":"rgba(248,118,109,1)","dash":"solid"},"hoveron":"points","name":"F","legendgroup":"F","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[1896,1900,1904,1906,1908,1912,1920,1924,1928,1932,1936,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1994,1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,2016],"y":[0,4,1,1,5,10,6,4,7,6,7,4,5,2,7,3,3,10,4,7,11,7,5,2,4,4,6,1,6,3,8,2,8,2,9],"text":["Sex: M<br />Sex: M<br />Year: 1896<br />gold_number:  0","Sex: M<br />Sex: M<br />Year: 1900<br />gold_number:  4","Sex: M<br />Sex: M<br />Year: 1904<br />gold_number:  1","Sex: M<br />Sex: M<br />Year: 1906<br />gold_number:  1","Sex: M<br />Sex: M<br />Year: 1908<br />gold_number:  5","Sex: M<br />Sex: M<br />Year: 1912<br />gold_number: 10","Sex: M<br />Sex: M<br />Year: 1920<br />gold_number:  6","Sex: M<br />Sex: M<br />Year: 1924<br />gold_number:  4","Sex: M<br />Sex: M<br />Year: 1928<br />gold_number:  7","Sex: M<br />Sex: M<br />Year: 1932<br />gold_number:  6","Sex: M<br />Sex: M<br />Year: 1936<br />gold_number:  7","Sex: M<br />Sex: M<br />Year: 1948<br />gold_number:  4","Sex: M<br />Sex: M<br />Year: 1952<br />gold_number:  5","Sex: M<br />Sex: M<br />Year: 1956<br />gold_number:  2","Sex: M<br />Sex: M<br />Year: 1960<br />gold_number:  7","Sex: M<br />Sex: M<br />Year: 1964<br />gold_number:  3","Sex: M<br />Sex: M<br />Year: 1968<br />gold_number:  3","Sex: M<br />Sex: M<br />Year: 1972<br />gold_number: 10","Sex: M<br />Sex: M<br />Year: 1976<br />gold_number:  4","Sex: M<br />Sex: M<br />Year: 1980<br />gold_number:  7","Sex: M<br />Sex: M<br />Year: 1984<br />gold_number: 11","Sex: M<br />Sex: M<br />Year: 1988<br />gold_number:  7","Sex: M<br />Sex: M<br />Year: 1992<br />gold_number:  5","Sex: M<br />Sex: M<br />Year: 1994<br />gold_number:  2","Sex: M<br />Sex: M<br />Year: 1996<br />gold_number:  4","Sex: M<br />Sex: M<br />Year: 1998<br />gold_number:  4","Sex: M<br />Sex: M<br />Year: 2000<br />gold_number:  6","Sex: M<br />Sex: M<br />Year: 2002<br />gold_number:  1","Sex: M<br />Sex: M<br />Year: 2004<br />gold_number:  6","Sex: M<br />Sex: M<br />Year: 2006<br />gold_number:  3","Sex: M<br />Sex: M<br />Year: 2008<br />gold_number:  8","Sex: M<br />Sex: M<br />Year: 2010<br />gold_number:  2","Sex: M<br />Sex: M<br />Year: 2012<br />gold_number:  8","Sex: M<br />Sex: M<br />Year: 2014<br />gold_number:  2","Sex: M<br />Sex: M<br />Year: 2016<br />gold_number:  9"],"type":"scatter","mode":"lines","line":{"width":1.88976377952756,"color":"rgba(0,191,196,1)","dash":"solid"},"hoveron":"points","name":"M","legendgroup":"M","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[1908,1920,1924,1928,1932,1936,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1994,1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,2016],"y":[0,0,0,0,0,0,0,0,0,2,2,0,0,2,2,9,3,2,0,3,1,5,5,4,3,3,4,4,2,4],"text":["Sex: F<br />Year: 1908<br />gold_number:  0","Sex: F<br />Year: 1920<br />gold_number:  0","Sex: F<br />Year: 1924<br />gold_number:  0","Sex: F<br />Year: 1928<br />gold_number:  0","Sex: F<br />Year: 1932<br />gold_number:  0","Sex: F<br />Year: 1936<br />gold_number:  0","Sex: F<br />Year: 1948<br />gold_number:  0","Sex: F<br />Year: 1952<br />gold_number:  0","Sex: F<br />Year: 1956<br />gold_number:  0","Sex: F<br />Year: 1960<br />gold_number:  2","Sex: F<br />Year: 1964<br />gold_number:  2","Sex: F<br />Year: 1968<br />gold_number:  0","Sex: F<br />Year: 1972<br />gold_number:  0","Sex: F<br />Year: 1976<br />gold_number:  2","Sex: F<br />Year: 1980<br />gold_number:  2","Sex: F<br />Year: 1984<br />gold_number:  9","Sex: F<br />Year: 1988<br />gold_number:  3","Sex: F<br />Year: 1992<br />gold_number:  2","Sex: F<br />Year: 1994<br />gold_number:  0","Sex: F<br />Year: 1996<br />gold_number:  3","Sex: F<br />Year: 1998<br />gold_number:  1","Sex: F<br />Year: 2000<br />gold_number:  5","Sex: F<br />Year: 2002<br />gold_number:  5","Sex: F<br />Year: 2004<br />gold_number:  4","Sex: F<br />Year: 2006<br />gold_number:  3","Sex: F<br />Year: 2008<br />gold_number:  3","Sex: F<br />Year: 2010<br />gold_number:  4","Sex: F<br />Year: 2012<br />gold_number:  4","Sex: F<br />Year: 2014<br />gold_number:  2","Sex: F<br />Year: 2016<br />gold_number:  4"],"type":"scatter","mode":"markers","marker":{"autocolorscale":false,"color":"rgba(248,118,109,1)","opacity":1,"size":7.55905511811024,"symbol":"circle","line":{"width":1.88976377952756,"color":"rgba(248,118,109,1)"}},"hoveron":"points","name":"F","legendgroup":"F","showlegend":false,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"x":[1896,1900,1904,1906,1908,1912,1920,1924,1928,1932,1936,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1994,1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,2016],"y":[0,4,1,1,5,10,6,4,7,6,7,4,5,2,7,3,3,10,4,7,11,7,5,2,4,4,6,1,6,3,8,2,8,2,9],"text":["Sex: M<br />Year: 1896<br />gold_number:  0","Sex: M<br />Year: 1900<br />gold_number:  4","Sex: M<br />Year: 1904<br />gold_number:  1","Sex: M<br />Year: 1906<br />gold_number:  1","Sex: M<br />Year: 1908<br />gold_number:  5","Sex: M<br />Year: 1912<br />gold_number: 10","Sex: M<br />Year: 1920<br />gold_number:  6","Sex: M<br />Year: 1924<br />gold_number:  4","Sex: M<br />Year: 1928<br />gold_number:  7","Sex: M<br />Year: 1932<br />gold_number:  6","Sex: M<br />Year: 1936<br />gold_number:  7","Sex: M<br />Year: 1948<br />gold_number:  4","Sex: M<br />Year: 1952<br />gold_number:  5","Sex: M<br />Year: 1956<br />gold_number:  2","Sex: M<br />Year: 1960<br />gold_number:  7","Sex: M<br />Year: 1964<br />gold_number:  3","Sex: M<br />Year: 1968<br />gold_number:  3","Sex: M<br />Year: 1972<br />gold_number: 10","Sex: M<br />Year: 1976<br />gold_number:  4","Sex: M<br />Year: 1980<br />gold_number:  7","Sex: M<br />Year: 1984<br />gold_number: 11","Sex: M<br />Year: 1988<br />gold_number:  7","Sex: M<br />Year: 1992<br />gold_number:  5","Sex: M<br />Year: 1994<br />gold_number:  2","Sex: M<br />Year: 1996<br />gold_number:  4","Sex: M<br />Year: 1998<br />gold_number:  4","Sex: M<br />Year: 2000<br />gold_number:  6","Sex: M<br />Year: 2002<br />gold_number:  1","Sex: M<br />Year: 2004<br />gold_number:  6","Sex: M<br />Year: 2006<br />gold_number:  3","Sex: M<br />Year: 2008<br />gold_number:  8","Sex: M<br />Year: 2010<br />gold_number:  2","Sex: M<br />Year: 2012<br />gold_number:  8","Sex: M<br />Year: 2014<br />gold_number:  2","Sex: M<br />Year: 2016<br />gold_number:  9"],"type":"scatter","mode":"markers","marker":{"autocolorscale":false,"color":"rgba(0,191,196,1)","opacity":1,"size":7.55905511811024,"symbol":"circle","line":{"width":1.88976377952756,"color":"rgba(0,191,196,1)"}},"hoveron":"points","name":"M","legendgroup":"M","showlegend":false,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null}],"layout":{"margin":{"t":43.7625570776256,"r":7.30593607305936,"b":40.1826484018265,"l":31.4155251141553},"plot_bgcolor":"rgba(235,235,235,1)","paper_bgcolor":"rgba(255,255,255,1)","font":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"title":"USA's Gold Medal Number by Sex","titlefont":{"color":"rgba(0,0,0,1)","family":"","size":17.5342465753425},"xaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[1890,2022],"tickmode":"array","ticktext":["1890","1920","1950","1980","2010"],"tickvals":[1890,1920,1950,1980,2010],"categoryorder":"array","categoryarray":["1890","1920","1950","1980","2010"],"nticks":null,"ticks":"outside","tickcolor":"rgba(51,51,51,1)","ticklen":3.65296803652968,"tickwidth":0.66417600664176,"showticklabels":true,"tickfont":{"color":"rgba(77,77,77,1)","family":"","size":11.689497716895},"tickangle":-0,"showline":false,"linecolor":null,"linewidth":0,"showgrid":true,"gridcolor":"rgba(255,255,255,1)","gridwidth":0.66417600664176,"zeroline":false,"anchor":"y","title":"Year","titlefont":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"hoverformat":".2f"},"yaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[-0.55,11.55],"tickmode":"array","ticktext":["0","3","6","9"],"tickvals":[0,3,6,9],"categoryorder":"array","categoryarray":["0","3","6","9"],"nticks":null,"ticks":"outside","tickcolor":"rgba(51,51,51,1)","ticklen":3.65296803652968,"tickwidth":0.66417600664176,"showticklabels":true,"tickfont":{"color":"rgba(77,77,77,1)","family":"","size":11.689497716895},"tickangle":-0,"showline":false,"linecolor":null,"linewidth":0,"showgrid":true,"gridcolor":"rgba(255,255,255,1)","gridwidth":0.66417600664176,"zeroline":false,"anchor":"x","title":"Gold Medal Number","titlefont":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"hoverformat":".2f"},"shapes":[{"type":"rect","fillcolor":null,"line":{"color":null,"width":0,"linetype":[]},"yref":"paper","xref":"paper","x0":0,"x1":1,"y0":0,"y1":1}],"showlegend":true,"legend":{"bgcolor":"rgba(255,255,255,1)","bordercolor":"transparent","borderwidth":1.88976377952756,"font":{"color":"rgba(0,0,0,1)","family":"","size":11.689497716895},"y":0.913385826771654},"annotations":[{"text":"Gender","x":1.02,"y":1,"showarrow":false,"ax":0,"ay":0,"font":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"xref":"paper","yref":"paper","textangle":-0,"xanchor":"left","yanchor":"bottom","legendTitle":true}],"hovermode":"closest","barmode":"relative"},"config":{"doubleClick":"reset","cloud":false},"source":"A","attrs":{"7c059f123fe":{"colour":{},"x":{},"y":{},"type":"scatter"},"7c0667226d3":{"colour":{},"x":{},"y":{}}},"cur_data":"7c059f123fe","visdat":{"7c059f123fe":["function (y) ","x"],"7c0667226d3":["function (y) ","x"]},"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script><!--/html_preserve-->

```r
## In this interation graph, readers can easily get the statistic of gold number and year by pointing to each spot.
interation_2 = g + geom_bar(aes(weight=medal_number),position = "dodge")+scale_fill_manual(values=c("darkred", "gold","gray40"))+ggtitle("Top 5 Medal winning NOCs")+labs(x="NOC",
         y="Medal Number") 
ggplotly(interation_2)
```

<!--html_preserve--><div id="htmlwidget-4f8d300cc120d58c95fe" style="width:672px;height:480px;" class="plotly html-widget"></div>
<script type="application/json" data-for="htmlwidget-4f8d300cc120d58c95fe">{"x":{"data":[{"orientation":"v","width":[0.3,0.3,0.3,0.3,0.300000000000001],"base":[0,0,0,0,0],"x":[0.7,1.7,2.7,3.7,4.7],"y":[587,620,649,596,1197],"text":["medal_number:  587<br />count:  587<br />NOC: FRA<br />Medal: Bronze","medal_number:  620<br />count:  620<br />NOC: GBR<br />Medal: Bronze","medal_number:  649<br />count:  649<br />NOC: GER<br />Medal: Bronze","medal_number:  596<br />count:  596<br />NOC: URS<br />Medal: Bronze","medal_number: 1197<br />count: 1197<br />NOC: USA<br />Medal: Bronze"],"type":"bar","marker":{"autocolorscale":false,"color":"rgba(139,0,0,1)","line":{"width":1.88976377952756,"color":"transparent"}},"name":"Bronze","legendgroup":"Bronze","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"orientation":"v","width":[0.3,0.3,0.3,0.3,0.300000000000001],"base":[0,0,0,0,0],"x":[1,2,3,4,5],"y":[463,635,592,832,2472],"text":["medal_number:  463<br />count:  463<br />NOC: FRA<br />Medal: Gold","medal_number:  635<br />count:  635<br />NOC: GBR<br />Medal: Gold","medal_number:  592<br />count:  592<br />NOC: GER<br />Medal: Gold","medal_number:  832<br />count:  832<br />NOC: URS<br />Medal: Gold","medal_number: 2472<br />count: 2472<br />NOC: USA<br />Medal: Gold"],"type":"bar","marker":{"autocolorscale":false,"color":"rgba(255,215,0,1)","line":{"width":1.88976377952756,"color":"transparent"}},"name":"Gold","legendgroup":"Gold","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null},{"orientation":"v","width":[0.3,0.3,0.3,0.300000000000001,0.300000000000001],"base":[0,0,0,0,0],"x":[1.3,2.3,3.3,4.3,5.3],"y":[567,729,538,635,1333],"text":["medal_number:  567<br />count:  567<br />NOC: FRA<br />Medal: Silver","medal_number:  729<br />count:  729<br />NOC: GBR<br />Medal: Silver","medal_number:  538<br />count:  538<br />NOC: GER<br />Medal: Silver","medal_number:  635<br />count:  635<br />NOC: URS<br />Medal: Silver","medal_number: 1333<br />count: 1333<br />NOC: USA<br />Medal: Silver"],"type":"bar","marker":{"autocolorscale":false,"color":"rgba(102,102,102,1)","line":{"width":1.88976377952756,"color":"transparent"}},"name":"Silver","legendgroup":"Silver","showlegend":true,"xaxis":"x","yaxis":"y","hoverinfo":"text","frame":null}],"layout":{"margin":{"t":43.7625570776256,"r":7.30593607305936,"b":40.1826484018265,"l":48.9497716894977},"plot_bgcolor":"rgba(235,235,235,1)","paper_bgcolor":"rgba(255,255,255,1)","font":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"title":"Top 5 Medal winning NOCs","titlefont":{"color":"rgba(0,0,0,1)","family":"","size":17.5342465753425},"xaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[0.4,5.6],"tickmode":"array","ticktext":["FRA","GBR","GER","URS","USA"],"tickvals":[1,2,3,4,5],"categoryorder":"array","categoryarray":["FRA","GBR","GER","URS","USA"],"nticks":null,"ticks":"outside","tickcolor":"rgba(51,51,51,1)","ticklen":3.65296803652968,"tickwidth":0.66417600664176,"showticklabels":true,"tickfont":{"color":"rgba(77,77,77,1)","family":"","size":11.689497716895},"tickangle":-0,"showline":false,"linecolor":null,"linewidth":0,"showgrid":true,"gridcolor":"rgba(255,255,255,1)","gridwidth":0.66417600664176,"zeroline":false,"anchor":"y","title":"NOC","titlefont":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"hoverformat":".2f"},"yaxis":{"domain":[0,1],"automargin":true,"type":"linear","autorange":false,"range":[-123.6,2595.6],"tickmode":"array","ticktext":["0","500","1000","1500","2000","2500"],"tickvals":[0,500,1000,1500,2000,2500],"categoryorder":"array","categoryarray":["0","500","1000","1500","2000","2500"],"nticks":null,"ticks":"outside","tickcolor":"rgba(51,51,51,1)","ticklen":3.65296803652968,"tickwidth":0.66417600664176,"showticklabels":true,"tickfont":{"color":"rgba(77,77,77,1)","family":"","size":11.689497716895},"tickangle":-0,"showline":false,"linecolor":null,"linewidth":0,"showgrid":true,"gridcolor":"rgba(255,255,255,1)","gridwidth":0.66417600664176,"zeroline":false,"anchor":"x","title":"Medal Number","titlefont":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"hoverformat":".2f"},"shapes":[{"type":"rect","fillcolor":null,"line":{"color":null,"width":0,"linetype":[]},"yref":"paper","xref":"paper","x0":0,"x1":1,"y0":0,"y1":1}],"showlegend":true,"legend":{"bgcolor":"rgba(255,255,255,1)","bordercolor":"transparent","borderwidth":1.88976377952756,"font":{"color":"rgba(0,0,0,1)","family":"","size":11.689497716895},"y":0.913385826771654},"annotations":[{"text":"Medal","x":1.02,"y":1,"showarrow":false,"ax":0,"ay":0,"font":{"color":"rgba(0,0,0,1)","family":"","size":14.6118721461187},"xref":"paper","yref":"paper","textangle":-0,"xanchor":"left","yanchor":"bottom","legendTitle":true}],"hovermode":"closest","barmode":"relative"},"config":{"doubleClick":"reset","cloud":false},"source":"A","attrs":{"7c0519e77ca":{"weight":{},"x":{},"fill":{},"type":"bar"}},"cur_data":"7c0519e77ca","visdat":{"7c0519e77ca":["function (y) ","x"]},"highlight":{"on":"plotly_click","persistent":false,"dynamic":false,"selectize":false,"opacityDim":0.2,"selected":{"opacity":1},"debounce":0},"shinyEvents":["plotly_hover","plotly_click","plotly_selected","plotly_relayout","plotly_brushed","plotly_brushing","plotly_clickannotation","plotly_doubleclick","plotly_deselect","plotly_afterplot"],"base_url":"https://plot.ly"},"evals":[],"jsHooks":[]}</script><!--/html_preserve-->

```r
# In this interation graph, readers can easily read the specific number of medals by pointing to each bar.
```
##6.Data Table

```r
#install.packages('DT')
library(DT)
datatable(cleandata_add_Medal_top5)
```

<!--html_preserve--><div id="htmlwidget-56f899378a94ecf6ec0f" style="width:100%;height:auto;" class="datatables html-widget"></div>
<script type="application/json" data-for="htmlwidget-56f899378a94ecf6ec0f">{"x":{"filter":"none","data":[["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"],["FRA","FRA","FRA","GBR","GBR","GBR","GER","GER","GER","URS","URS","URS","USA","USA","USA"],["Bronze","Gold","Silver","Bronze","Gold","Silver","Bronze","Gold","Silver","Bronze","Gold","Silver","Bronze","Gold","Silver"],[587,463,567,620,635,729,649,592,538,596,832,635,1197,2472,1333]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th> <\/th>\n      <th>NOC<\/th>\n      <th>Medal<\/th>\n      <th>medal_number<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"columnDefs":[{"className":"dt-right","targets":3},{"orderable":false,"targets":0}],"order":[],"autoWidth":false,"orderClasses":false}},"evals":[],"jsHooks":[]}</script><!--/html_preserve-->

```r
library(stringr)
pretty_headers <- 
  gsub("[.]", " ", colnames(cleandata_add_Medal_top5)) %>%
  str_to_title()
cleandata_add_Medal_top5 %>%
  datatable(
    rownames = FALSE,
    colnames = pretty_headers,
    filter = list(position = "top"),
    options = list(language = list(sSearch = "Filter:"))
  )
```

<!--html_preserve--><div id="htmlwidget-46592b4fd45987c39f22" style="width:100%;height:auto;" class="datatables html-widget"></div>
<script type="application/json" data-for="htmlwidget-46592b4fd45987c39f22">{"x":{"filter":"top","filterHTML":"<tr>\n  <td data-type=\"factor\" style=\"vertical-align: top;\">\n    <div class=\"form-group has-feedback\" style=\"margin-bottom: auto;\">\n      <input type=\"search\" placeholder=\"All\" class=\"form-control\" style=\"width: 100%;\"/>\n      <span class=\"glyphicon glyphicon-remove-circle form-control-feedback\"><\/span>\n    <\/div>\n    <div style=\"width: 100%; display: none;\">\n      <select multiple=\"multiple\" style=\"width: 100%;\" data-options=\"[&quot;FRA&quot;,&quot;GBR&quot;,&quot;GER&quot;,&quot;URS&quot;,&quot;USA&quot;]\"><\/select>\n    <\/div>\n  <\/td>\n  <td data-type=\"factor\" style=\"vertical-align: top;\">\n    <div class=\"form-group has-feedback\" style=\"margin-bottom: auto;\">\n      <input type=\"search\" placeholder=\"All\" class=\"form-control\" style=\"width: 100%;\"/>\n      <span class=\"glyphicon glyphicon-remove-circle form-control-feedback\"><\/span>\n    <\/div>\n    <div style=\"width: 100%; display: none;\">\n      <select multiple=\"multiple\" style=\"width: 100%;\" data-options=\"[&quot;Bronze&quot;,&quot;Gold&quot;,&quot;Silver&quot;]\"><\/select>\n    <\/div>\n  <\/td>\n  <td data-type=\"integer\" style=\"vertical-align: top;\">\n    <div class=\"form-group has-feedback\" style=\"margin-bottom: auto;\">\n      <input type=\"search\" placeholder=\"All\" class=\"form-control\" style=\"width: 100%;\"/>\n      <span class=\"glyphicon glyphicon-remove-circle form-control-feedback\"><\/span>\n    <\/div>\n    <div style=\"display: none; position: absolute; width: 200px;\">\n      <div data-min=\"463\" data-max=\"2472\"><\/div>\n      <span style=\"float: left;\"><\/span>\n      <span style=\"float: right;\"><\/span>\n    <\/div>\n  <\/td>\n<\/tr>","data":[["FRA","FRA","FRA","GBR","GBR","GBR","GER","GER","GER","URS","URS","URS","USA","USA","USA"],["Bronze","Gold","Silver","Bronze","Gold","Silver","Bronze","Gold","Silver","Bronze","Gold","Silver","Bronze","Gold","Silver"],[587,463,567,620,635,729,649,592,538,596,832,635,1197,2472,1333]],"container":"<table class=\"display\">\n  <thead>\n    <tr>\n      <th>Noc<\/th>\n      <th>Medal<\/th>\n      <th>Medal_number<\/th>\n    <\/tr>\n  <\/thead>\n<\/table>","options":{"language":{"sSearch":"Filter:"},"columnDefs":[{"className":"dt-right","targets":2}],"order":[],"autoWidth":false,"orderClasses":false,"orderCellsTop":true}},"evals":[],"jsHooks":[]}</script><!--/html_preserve-->

```r
## In this datatable, I can provide the medal information for a particular NOC(by using the column filter of Noc), I can also provide how gold medals are distributed in the top 5 Medal Winning NOCs(by using the column filter of Medal)
```

