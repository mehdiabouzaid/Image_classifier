
% add the required search paths
setup ;

% --------------------------------------------------------------------
% Stage A: Data Preparation
% --------------------------------------------------------------------

% Load training data
pos = load('data/motorbike_train_hist.mat') ;
% ---- FOR QUESTION 6: uncomment the appropriate line ----
%pos = load('data/person_train_hist.mat') ;
%pos = load('data/aeroplane_train_hist.mat') ;
neg = load('data/background_train_hist.mat') ;
names = {pos.names{:}, neg.names{:}};
histograms = [pos.histograms, neg.histograms] ;
labels = [ones(1,numel(pos.names)) -ones(1,numel(neg.names))] ;
clear pos neg ;

% Load testing data
pos = load('data/motorbike_val_hist.mat') ;
% --- FOR QUESTION 6: uncomment the appropriate line ---
%pos = load('data/person_val_hist.mat') ;
%pos = load('data/aeroplane_val_hist.mat') ;
neg = load('data/background_val_hist.mat') ;
testNames = {pos.names{:}, neg.names{:}};
testHistograms = [pos.histograms, neg.histograms] ;
testLabels = [ones(1,numel(pos.names)) -ones(1,numel(neg.names))] ;
clear pos neg ;


% count how many images are there
fprintf('Number of training images: %d positive, %d negative\n', ...
        sum(labels > 0), sum(labels < 0)) ;
fprintf('Number of testing images: %d positive, %d negative\n', ...
        sum(testLabels > 0), sum(testLabels < 0)) ;

% For QUESTION 4: Vary the image representation
% Uncomment the lines below to analyze the influence of representation
%histograms = removeSpatialInformation(histograms) ;
%testHistograms = removeSpatialInformation(testHistograms) ;

% For QUESTION 5: Vary the classifier (Hellinger kernel)
% ---------- A COMPLETER ---------------
% ** insert code here for the Hellinger kernel using  **
% ** the training histograms and testHistograms       **


% L2 normalize the histograms before running the linear SVM
histograms = bsxfun(@times, histograms, 1./sqrt(sum(histograms.^2,1))) ;
testHistograms = bsxfun(@times, testHistograms, 1./sqrt(sum(testHistograms.^2,1))) ;


% --------------------------------------------------------------------
% Stage B: Training a classifier
% --------------------------------------------------------------------

% Train the linear SVM. The SVM paramter C has to be cross-validated.
% --------- A COMPLETER : SELECTION DE C  -----------
% insert here code for C parameter cross-validation
C = 100 ;
% For speed and efficiency purpose, we provide an enhanced implementation of linear SVM (based on online learnin)
[w, bias] = trainLinearSVM(histograms, labels, C) ;

% Visualize visual words by relevance based on vector w on the first image
% uncomment the line below to visualise
 %displayRelevantVisualWords(names{1},w)

% Evaluate the linear SVM  on the training data
scores = w' * histograms + bias ;

% Visualize the ranked list of images
% uncomment lines below to visualize images deemed most relevant to positives category
 figure(1) ; clf ; set(1,'name','Ranked training images (subset)') ;
 displayRankedImageList(names, scores)  ;


% ---- A COMPLETER TRACE COURBE ROC  ------------
% insert here code to display the ROC curve for training data
figure(2) ; clf ; set(2,'name','Precision-recall on train data') ;
vl_roc(labels, scores) ;

% -------- A COMPLETER : PERFORMANCES EN APPRENTISSAGE : Matrice de Confusion, taux d'erreur -------
% insert here code to compute confusion matrice, classification accuracy ...


% --------------------------------------------------------------------
% Stage C: Classify the test images and assess the performance
% --------------------------------------------------------------------

% ---------- A COMPLETER --------------
% insert here code to test the linear SVM on test dataset
 testScores = w' * testHistograms + bias ;

% Visualize the ranked list of images
% uncomment lines below to visualize test images deemed most relevant to positives category
figure(3) ; clf ; set(3,'name','Ranked test images (subset)') ;
displayRankedImageList(testNames, testScores)  ; % testScores in the scores f(x) provided by SVM on the test set


% --------- A COMPLETER :  TRACE COURBE ROC  ------------
% insert here code to display the ROC curve for test data
figure(4) ; clf ; set(4,'name','Precision-recall on test data') ;
vl_roc(testLabels, testScores) ;


% -------- A COMPLETER : PERFORMANCES EN APPRENTISSAGE : Matrice de Confusion, taux d'erreur -------
% insert here code to compute confusion matrice, classification accuracy on test data ...
[drop,drop,info] = vl_roc(testLabels, testScores) ;
fprintf('Test AP: %.2f\n', info.auc) ;

[drop,perm] = sort(testScores,'descend') ;
fprintf('Correctly retrieved in the top 36: %d\n', sum(testLabels(perm(1:36)) > 0)) ;

