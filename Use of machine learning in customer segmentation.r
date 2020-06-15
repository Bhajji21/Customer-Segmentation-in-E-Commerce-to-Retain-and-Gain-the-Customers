
customer_data = read.csv("C:/Users/abhay/OneDrive/Desktop/customer-segmentation-dataset/Mall_Customers.csv")

str(customer_data)

names(customer_data)

head(customer_data)

summary(customer_data$Age)

summary(customer_data$Annual.Income)

sd(customer_data$Age)

sd(customer_data$Annual.Income)

summary(customer_data$Age)

sd(customer_data$Spending.Score)

#customer Gender visualization
cus_gender_vis=table(customer_data$Gender)
barplot(a,main="Using BarPlot to display Gender Comparision",
       ylab="Count",
       xlab="Gender",
       col=rainbow(2),
       legend=rownames(cus_gender_vis))

#Ratio of male and female through piechart
install.packages('plotrix')
piechart=round(a/sum(a)*100)
lbs=paste(c("Female","Male")," ",piechart,"%",sep=" ")
library(plotrix)
pie3D(a,labels=lbs,
   main="Pie Chart Depicting Ratio of Female and Male")

#Visualization based on Age
hist(customer_data$Age,
    col="green",
    main="Histogram to Show Count of Age Class",
    xlab="Age Class",
    ylab="Frequency",
    labels=TRUE)

#boxplot visualization to check the range of age
boxplot(customer_data$Age,
       col="yellow",
       main="Descriptive Analysis of Age through Boxplot")

#analysis of annual income of customers
hist(customer_data$Annual.Income,
  col="red",
  main="Histogram for Annual Income",
  xlab="Annual Income Class",
  ylab="Frequency",
  labels=TRUE)

#density plot
plot(density(customer_data$Annual.Income),
    col="yellow",
    main="Density Plot for Annual Income of customers",
    xlab="Annual Income Class",
    ylab="Density")
polygon(density(customer_data$Annual.Income..k..),
        col="blue")

#Analysing spending scores of customers
summary(customer_data$Spending.Score)

hist(customer_data$Spending.Score,
  col="black",
  main="Histogram for Spending Score",
  xlab="Spending Score Class",
  ylab="Frequency",
  labels=TRUE)

#boxplot for spending scores
boxplot(customer_data$Spending.Score,
   horizontal=TRUE,
   col="orange",
   main="BoxPlot for Descriptive Analysis of Spending Score")

#machine learning models
#k-means algorithm
#For value of k
# Elbow method
install.packages('purrr')
library(purrr)
set.seed(123)
# function to calculate total intra-cluster sum of square 
iss <- function(k) {
  kmeans(customer_data[,3:5],k,iter.max=100,nstart=100,algorithm="Lloyd" )$tot.withinss
}
k.values <- 1:10
iss_values <- map_dbl(k.values, iss)
plot(k.values, iss_values,
    type="b", pch = 19, frame = FALSE, 
    xlab="Number of clusters K for given dataset",
    ylab="Total intra-clusters sum of squares")

# From above graph we can conclude that 4 is appropriate value of K
#Average Silhouette Method
#installing package cluster
install.packages('cluster')
library(cluster)

#installing package gridExtra
install.packages('gridExtra')
library(gridExtra)

#installing package grid
install.packages('grid')
library(grid)

k2 <- kmeans(customer_data[,3:5],2,iter.max=100,nstart=50,algorithm="Lloyd")
s2 <- plot(silhouette(k2$cluster,dist(customer_data[,3:5],"euclidean")))

k3 <- kmeans(customer_data[,3:5],3,iter.max=100,nstart=50,algorithm="Lloyd")
s3 <- plot(silhouette(k3$cluster,dist(customer_data[,3:5],"euclidean")))

k4 <- kmeans(customer_data[,3:5],4,iter.max=100,nstart=50,algorithm="Lloyd")
s4 <- plot(silhouette(k4$cluster,dist(customer_data[,3:5],"euclidean")))

k5 <- kmeans(customer_data[,3:5],5,iter.max=100,nstart=50,algorithm="Lloyd")
s5 <- plot(silhouette(k5$cluster,dist(customer_data[,3:5],"euclidean")))

k6 <- kmeans(customer_data[,3:5],6,iter.max=100,nstart=50,algorithm="Lloyd")
s6 <- plot(silhouette(k6$cluster,dist(customer_data[,3:5],"euclidean")))

k7 <- kmeans(customer_data[,3:5],7,iter.max=100,nstart=50,algorithm="Lloyd")
s7 <- plot(silhouette(k7$cluster,dist(customer_data[,3:5],"euclidean")))

k8 <- kmeans(customer_data[,3:5],8,iter.max=100,nstart=50,algorithm="Lloyd")
s8 <- plot(silhouette(k8$cluster,dist(customer_data[,3:5],"euclidean")))

k9 <- kmeans(customer_data[,3:5],9,iter.max=100,nstart=50,algorithm="Lloyd")
s9 <- plot(silhouette(k9$cluster,dist(customer_data[,3:5],"euclidean")))

k10 <- kmeans(customer_data[,3:5],10,iter.max=100,nstart=50,algorithm="Lloyd")
s10 <- plot(silhouette(k10$cluster,dist(customer_data[,3:5],"euclidean")))

#installing and loading package ggplot2
install.packages('ggplot2')
library(ggplot2)

#installing and loading package NbClust
install.packages('NbClust')
library(NbClust)

#installing and loading package "FactoMineR"
install.packages("FactoMineR")
library(FactoMineR)

#installing and loading package faceoextra
install.packages("factoextra")
library(factoextra)

fviz_nbclust(customer_data[,3:5], kmeans, method = "silhouette")

#Gap Statistic method to calculate k
set.seed(125)
statistic_gap <- clusGap(customer_data[,3:5], FUN = kmeans, nstart = 25,
            K.max = 10, B = 50)
fviz_gap_stat(statistic_gap)

k6 <- kmeans(customer_data[,3:5],6,iter.max=100,nstart=50,algorithm="Lloyd")
k6

#From above available components
#cluster – vector denoting the cluster which has an allocation of each point.
#totss – representing the total sum of squares.
#centers – Matrix comprising of several cluster centers
#withinss – vector representing the intra-cluster sum of squares having one component per cluster.
#tot.withinss – denoting the total intra-cluster sum of squares.
#betweenss – representing the sum of between-cluster squares.
#size – total number of points that each cluster holds.

#Using two principle component visualize the clustering results

principle_component_clust <- prcomp(customer_data[,3:5],scale=FALSE)
summary(principle_component_clust)


#using first two principle component
principle_component_clust$rotation[,1:2]

#segmentation of customers for given data
#Visualization of clustering based on Annual income and spending score 
set.seed(1)
ggplot(customer_data, aes(x =Annual.Income..k.., y = Spending.Score..1.100.)) + 
  geom_point(stat = "identity", aes(color = as.factor(k6$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c("1", "2", "3", "4", "5","6"),
              labels=c("Customer_Cluster 1", "Customer_Cluster 2", "Customer_Cluster 3", "Customer_Cluster 4", "Customer_Cluster 5","Customer_Cluster 6")) +
  ggtitle("Segmentation of Customers for given Dataset", subtitle = "Using K-means Clustering Algorithm")

#segmentation of customers for given data
#Visualization of clustering based on Annual income and Age
set.seed(1)
ggplot(customer_data, aes(x = Spending.Score..1.100., y=Age)) + 
  geom_point(stat = "identity", aes(color = as.factor(k6$cluster))) +
  scale_color_discrete(name=" ",
                       breaks=c("1", "2", "3", "4", "5","6"),
              labels=c("Customer_Cluster 1", "Customer_Cluster 2", "Customer_Cluster 3", "Customer_Cluster 4", "Customer_Cluster 5","Customer_Cluster 6")) +
  ggtitle("Segmentation of Customers for given Dataset", subtitle = "Using K-means Clustering Algorithm")

kCols=function(vec){cols=rainbow (length (unique (vec)))
return (cols[as.numeric(as.factor(vec))])}


# K-means clusters
digCluster <- k6$cluster;
dignm <- as.character(digCluster);


plot(principle_component_clust$x[,1:2], col =kCols(digCluster),pch =19,xlab ="K-means",ylab="classes")
legend("topright",unique(dignm),fill=unique(kCols(digCluster)))




