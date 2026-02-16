#    Beginner's Guide to GLM and GLMM with R
#    Alain Zuur, Joseph M Hilbe, and Elena N Ieno

#    www.highstat.com

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.



##############################################################
#Set the working directory (on a Mac) and load the data
setwd("/Users/Highstat/applicat/HighlandStatistics/Books/BGS/GLM/Data/Ricardo")
PO <- read.table(file = "PolychaetaV3.txt",
                 header = TRUE)
str(PO)

#'data.frame':	144 obs. of  4 variables:
# $ Hsimilis: int  0 0 0 0 0 0 0 0 0 0 ...
# $ Level   : Factor w/ 2 levels "Lower","Upper": 1 1 1 1 1 1 2 1 1 1 ...
# $ Location: Factor w/ 3 levels "A","B","C": 1 1 1 1 1 1 1 1 1 1 ...
# $ MedSand : num  16.24 10.73 5.84 10.03 25.95 ...


###################################################################
#Load packages and library files
library(lattice)  #Needed for data exploration
library(mgcv)  # Needed for data exploration
library(coefplot2)# Needed for glm/JAGS comp
library(R2jags)  # MCMC
source(file = "/Users/Highstat/applicat/HighlandStatistics/Courses/FilesOnlineFollowUpRegressionGLMGAM/FinalExercises/HighstatLibV6.R")  
source(file = "/Users/Highstat/applicat/HighlandStatistics/MCMC/R/MCMCSupportHighstat.R")
##################################################################



##################################################################
#Data exploration
#Visualizing number of zeros and ones
table(PO$Hsimilis)

#Dotplot of the continuos covariates grouped by location
dotchart(PO$MedSand, xlab ="Medium sand values",
         ylab = "Order of the data",
         cex.lab = 1.5,
         groups = factor(PO$Location))


# To surpress the vertical dotted lines, I typed
# dotchart  in R, copied and pasted the 
# code and put a # in front of the abline function

dotchart2 <- function (x, labels = NULL, groups = NULL, gdata = NULL, cex = par("cex"), 
    pch = 21, gpch = 21, bg = par("bg"), color = par("fg"), gcolor = par("fg"), 
    lcolor = "gray", xlim = range(x[is.finite(x)]), main = NULL, 
    xlab = NULL, ylab = NULL, ...) 
{
    opar <- par("mai", "mar", "cex", "yaxs")
    on.exit(par(opar))
    par(cex = cex, yaxs = "i")
    if (!is.numeric(x)) 
        stop("'x' must be a numeric vector or matrix")
    n <- length(x)
    if (is.matrix(x)) {
        if (is.null(labels)) 
            labels <- rownames(x)
        if (is.null(labels)) 
            labels <- as.character(1L:nrow(x))
        labels <- rep_len(labels, n)
        if (is.null(groups)) 
            groups <- col(x, as.factor = TRUE)
        glabels <- levels(groups)
    }
    else {
        if (is.null(labels)) 
            labels <- names(x)
        glabels <- if (!is.null(groups)) 
            levels(groups)
        if (!is.vector(x)) {
            warning("'x' is neither a vector nor a matrix: using as.numeric(x)")
            x <- as.numeric(x)
        }
    }
    plot.new()
    linch <- if (!is.null(labels)) 
        max(strwidth(labels, "inch"), na.rm = TRUE)
    else 0
    if (is.null(glabels)) {
        ginch <- 0
        goffset <- 0
    }
    else {
        ginch <- max(strwidth(glabels, "inch"), na.rm = TRUE)
        goffset <- 0.4
    }
    if (!(is.null(labels) && is.null(glabels))) {
        nmai <- par("mai")
        nmai[2L] <- nmai[4L] + max(linch + goffset, ginch) + 
            0.1
        par(mai = nmai)
    }
    if (is.null(groups)) {
        o <- 1L:n
        y <- o
        ylim <- c(0, n + 1)
    }
    else {
        o <- sort.list(as.numeric(groups), decreasing = TRUE)
        x <- x[o]
        groups <- groups[o]
        color <- rep_len(color, length(groups))[o]
        lcolor <- rep_len(lcolor, length(groups))[o]
        offset <- cumsum(c(0, diff(as.numeric(groups)) != 0))
        y <- 1L:n + 2 * offset
        ylim <- range(0, y + 2)
    }
    plot.window(xlim = xlim, ylim = ylim, log = "")
    lheight <- par("csi")
    if (!is.null(labels)) {
        linch <- max(strwidth(labels, "inch"), na.rm = TRUE)
        loffset <- (linch + 0.1)/lheight
        labs <- labels[o]
        mtext(labs, side = 2, line = loffset, at = y, adj = 0, 
            col = color, las = 2, cex = cex, ...)
    }
    #abline(h = y, lty = "dotted", col = lcolor)  <------
    points(x, y, pch = pch, col = color, bg = bg)
    if (!is.null(groups)) {
        gpos <- rev(cumsum(rev(tapply(groups, groups, length)) + 
            2) - 1)
        ginch <- max(strwidth(glabels, "inch"), na.rm = TRUE)
        goffset <- (max(linch + 0.2, ginch, na.rm = TRUE) + 0.1)/lheight
        mtext(glabels, side = 2, line = goffset, at = gpos, adj = 0, 
            col = gcolor, las = 2, cex = cex, ...)
        if (!is.null(gdata)) {
            abline(h = gpos, lty = "dotted")
            points(gdata, gpos, pch = gpch, col = gcolor, bg = bg, 
                ...)
        }
    }
    axis(1)
    box()
    title(main = main, xlab = xlab, ylab = ylab, ...)
    invisible()
}


dotchart2(PO$MedSand, xlab ="Medium sand values",
         ylab = "Order of the data",
         cex.lab = 1.5,
         pch = 16,
         groups = factor(PO$Location))




#Relationships
#Presence-absence vs all covariates
xyplot(Hsimilis ~ MedSand | Level * Location,
       data = PO, pch = 16, col =1,
       strip = function(bg='white', ...) strip.default(bg='white', ...),
       scales = list(alternating = T,
                     x = list(relation = "free"),
                     y = list(relation = "same")),
       xlab = list(label = "Medium sand content (%)", cex = 1.5) ,
       ylab = list(label = "Presence/absence of H. similis", cex = 1.5)
  )


xyplot(Hsimilis ~ MedSand | Location,
       data = PO, pch = 16, col =1,
       layout = c(1, 3),
       strip = function(bg='white', ...) strip.default(bg='white', ...),
       scales = list(alternating = T,
                     x = list(relation = "free"),
                     y = list(relation = "same")),
       xlab = list(label = "Medium sand content (%)", cex = 1.5) ,
       ylab = list(label = "Presence/absence of H. similis", cex = 1.5)
  )


#Visualizing main terms  
plot.design(Hsimilis ~ Level + Location, data = PO)

#Number of obervations per level and location
table(PO$Location, PO$Level)
####################################################





####################################################
#Section 3.4
#Standarizing continuouis covariate
PO$MedSandC <- (PO$MedSand - mean(PO$MedSand))/sd(PO$MedSand)

#Running the GLM function
M1 <- glm(Hsimilis ~ MedSandC + Level + Location +
                       MedSandC : Level +
                       MedSandC : Location +
                       Level : Location , data = PO, family = binomial)
                       
summary(M1)
drop1(M1, test = "Chi")

#Model selection using step function
M1A <- glm(Hsimilis ~ 1, data = PO, family = binomial)


step(M1A, 
    scope = list(lower = ~1, 
                 upper =~ MedSandC + Level + Location +
                       MedSandC : Level +
                       MedSandC : Location +
                       Level : Location ))


#Results from the optimal model (using Level and Location)
M2 <- glm(Hsimilis ~ Level + Location,
                       data = PO, family = binomial)
summary(M2)
drop1(M2, test = "Chi")

######################################################
#Model validation

E2 <- resid(M2, type = "pearson")
F2 <- fitted(M2, type = "response")
plot(x = F2, y = E2, 
     xlab = "Fitted values", 
     ylab = "Pearson residuals",
     cex.lab = 1.5)
abline(h = 0, lty = 2)     

plot(cooks.distance(M2), 
     type = "h", 
     ylim = c(0,1),
     cex.lab = 1.5,
     ylab = "Cook distance values")

plot(x=PO$MedSand, y = E2,
     xlab = "Median sand content", 
     ylab = "Pearson residuals",
     cex.lab = 1.5)
abline(h = 0, lty = 2)     


######################################################
#Section 3.4.8 Visualizing the model
MyData <- expand.grid(Level = c("Lower", "Upper"),
                      Location  = c("A", "B", "C"))
X <- model.matrix(~Level + Location, data = MyData)

eta <- X %*% coef(M2)
MyData$pi <- exp(eta) / (1 + exp(eta))

SE <- sqrt(diag(X %*% vcov(M2) %*% t(X)))
MyData$SEup <- exp(eta + 2 * SE) / (1 + exp(eta + 2 * SE))
MyData$SElo <- exp(eta - 2 * SE) / (1 + exp(eta - 2 * SE))

MyX <- 1:6
MyXLab <- paste(MyData[,1], MyData[,2], sep = " ")
pr(mar = c(5,5,2,2))
plot(x = MyX, cex.lab = 1.5,
     y = MyData$pi,
     xlab = "Covariate combinations",
     ylab = "Predicted probabilities",
     axes = FALSE,
     type = "p",
     pch = 16,
     ylim = c(0, 1))
axis(2)
axis(1, at = MyX, labels = MyXLab )
box()
#F. Plot predicted values +/- 2* SE 
segments(MyX, MyData$SElo,
         MyX, MyData$SEup)          
######################################################



######################################################
#Section 3.5

#Specifying the data for JAGs
X <- model.matrix(~ MedSandC + Level + Location +
                       MedSandC : Level +
                       MedSandC : Location +
                       Level : Location, 
                    data = PO)
K <- ncol(X)

win.data <- list(Y    = PO$Hsimilis,
                 N    = nrow(PO),
                 X    = X,
                 K    = K,
                 LogN = log(nrow(PO))
                 )


#Jags modelling code
sink("GLM.txt")
cat("
model{
    #1. Priors
    for (i in 1:K) { beta[i]  ~ dnorm(0, 0.0001) }  
 
    #2. Likelihood 
    for (i in 1:N){  
      Y[i] ~ dbern(p[i])
      logit(p[i]) <- eta[i]
      eta[i]      <- inprod(beta[], X[i,])
      LLi[i] <- Y[i] * log(p[i]) +
                (1 - Y[i]) * log(1 - p[i])
  } 
  LogL <- sum(LLi[1:N])
  AIC <- -2 * LogL + 2 * K
  BIC <- -2 * LogL + LogN * K

}
",fill = TRUE)
sink()


#Set the initial values for the betas and sigma
inits <- function () {
  list(
    beta  = rnorm(K, 0, 0.1)
    )  }

#Parameters to estimate
params <- c("beta", "LogL", "AIC", "BIC")


######################################################
#Execute the JAGs code
J0 <- jags(data = win.data,
           inits = inits,
           parameters = params,
           model.file = "GLM.txt",
           n.thin = 10,
           n.chains = 3,
           n.burnin = 4000,
           n.iter   = 5000)
J1 <- update(J0, n.iter = 10000, n.thin = 10)
out <- J1$BUGSoutput

OUT1 <- MyBUGSOutput(out, c(uNames("beta", K), "LogL", "AIC", "BIC"))
print(OUT1, digits = 3)

#Compare output from JAGS to that of GLM in R
cbind(coef(M1),OUT1[1:K,1])
########################################################






########################################################
#Section 3.6 Model selection using AIC, DIC and BIC in JAGS

out
min(out$sims.list$AIC)


# Next step is to calculate the AIC, BIC or DIC for each potential model
# Or use it in a forward or backwards selection procedure
# therefore we need to adjust the matrix X and re run JAGs code each time
# We will present the results of the model with Location and Level.

X <- model.matrix(~ Level + Location, 
                    data = PO)
K <- ncol(X)
#Now rerun the win.data and jags code


print(OUT1, digits = 3)

#Compared results with M2
cbind(coef(M2),OUT1[1:K,1])

# Make a graph to compare these parameters using
# the frequentist approach vs MCMC results
# Load Coefplot2


beta1 <- coef(M2) #M2 results
se1   <- sqrt(diag(vcov(M2)))#M2 results


beta2 <- OUT1[1:K,1]# JAGS results
se2   <- OUT1[1:K,2] # JAGS results


coefplot2(beta1, se1, offset = 0, col =1, 
          varnames = names(beta1), xlim = c(-6,5),
          cex.var = 1.1, main = "")

coefplot2(beta2, se2, offset = 0.25, col =1, 
          varnames = names(beta1), add = TRUE)
#Top ones are JAGS
#########################################################




#########################################################
#Section 3.7 Model Interpretation
# Calculate 95% credible intervals inside JAGS
# Make a graph

X <- model.matrix(~ Level + Location,
                    data = PO)
K <- ncol(X)


MyData <- expand.grid(Level = c("Lower", "Upper"),
                      Location  = c("A", "B", "C"))
Xp <- model.matrix(~Level + Location, data = MyData)

# Matrices X and Xp are passed on to JAGs


win.data <- list(Y    = PO$Hsimilis,
                 N    = nrow(PO),
                 X    = X,
                 K    = K,
                 Xp   = Xp,
                 LogN = log(nrow(PO))
                 )

sink("GLM.txt")
cat("
model{
    #1. Priors
    for (i in 1:K) {beta[i] ~ dnorm(0, 0.0001) }

    #2. Likelihood
    for (i in 1:N){
      Y[i] ~ dbern(p[i])
      logit(p[i]) <- eta[i]
      eta[i]      <- inprod(beta[], X[i,])
      l[i] <- Y[i] * log(p[i]) + (1 - Y[i]) * log(1 - p[i])
  }
  L <- sum(l[1:N])
  AIC <- -2 * L + 2 * K
  BIC <- -2 * L + LogN * K

  #Predict probabilities
  for (j in 1:6){
    etaP[j]      <- inprod(beta[], Xp[j,])
    logit(pi[j]) <- etaP[j]
  }
}
",fill = TRUE)
sink()


#Use the same initial values for the betas and sigma
#from the previous section

#Parameters to estimate
params <- c("beta", "L", "AIC", "BIC", "pi")


######################################################
#Execute the code

J0 <- jags(data = win.data,
           inits = inits,
           parameters = params,
           model.file = "GLM.txt",
           n.thin = 10,
           n.chains = 3,
           n.burnin = 4000,
           n.iter   = 5000)

J1.upd <- update(J0, n.iter=10000, n.thin = 10)
out <- J1.upd$BUGSoutput
OUT1 <- MyBUGSOutput(out, c(uNames("pi", 6)))

#Code to plot the posterior means and 95% credible intervals
MyX <- 1:6
MyXLab <- paste(MyData[,1], MyData[,2], sep = " ")
par(mar = c(5,5,2,2))
plot(x = MyX, cex.lab = 1.5,
     y = OUT1[,1],
     xlab = "Covariate combinations",
     ylab = "Predicted probabilities",
     axes = FALSE,
     type = "p",
     pch = 16,
     ylim = c(0, 1))
axis(2)
axis(1, at = MyX, labels = MyXLab )
box()
#F. Plot predicted values +/- 2* SE
segments(MyX, OUT1[,3],
         MyX, OUT1[,4])

##################################################



#####################################################
#3.8 Discussion
#Why are there small differences?
#Simulate some data

set.seed(12345)
N <- 100
x <- sort(runif(N))
xm <- x - mean(x)
alpha <- 1
beta  <- 2
eta   <- alpha + beta*xm
pi    <- exp(eta) / (1 + exp(eta))
Y1 <- rbinom(N, size = 1, prob = pi) 

T1 <- glm(Y1 ~ xm,family = binomial)
summary(T1)           

    
X <- model.matrix(~ xm)    
X <- matrix(X, nrow = N)
K <- ncol(X)
win.data <- list(Y  = Y1,
                 N  = N,
                 X  = X,
                 K  = K,
                 LogN = log(N)
                 )

sink("GLMSim.txt")
cat("
model{
    #Priors
    for (i in 1:K) { beta[i] ~ dnorm(0, 0.0001) }  

    #######################
    #Likelihood 
    for (i in 1:N){  
      Y[i] ~ dbern(p[i])
      logit(p[i]) <- eta[i]
      eta[i]      <- inprod(beta[], X[i,])
      l[i]        <- Y[i] * log(p[i]) + (1 - Y[i]) * log(1 - p[i])
  } 
  L <- sum(l[1:N])
  AIC <- -2 * L + 2 * K
  BIC <- -2 * L + LogN * K
}
",fill = TRUE)
sink()


#Set the initial values for the betas and sigma
inits <- function () {
  list(
    beta  = rnorm(K, 0, 0.1)
    )  }

#Parameters to estimate
params <- c("beta", "L", "AIC", "BIC")


######################################################
J0 <- jags(data = win.data,
           inits = inits,
           parameters = params,
           model.file = "GLMSim.txt",
           n.thin = 100,
           n.chains = 3,
           n.burnin = 4000,
           n.iter   = 5000)

J1.upd <- update(J0, n.iter=10000, n.thin = 10)  
out <- J1.upd$BUGSoutput

min(out$sims.list$AIC)
AIC(M1)

min(out$sims.list$BIC)
BIC(M1)

print(out, digits = 3)
summary(T1)        
#Conclusion: For large N, results are identical
#            For small N up to 5% difference

#Conclusion from the actual analysis on the data:
#If the data is not good enough...then keep the model simple!

###############################END of CODE######################