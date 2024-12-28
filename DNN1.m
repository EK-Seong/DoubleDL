function [trainedModel, validationRMSE] = DNN1(trainingData)
% [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% 훈련된 회귀 모델과 그 RMSE을(를) 반환합니다. 이 코드는 회귀 학습기 앱에서 훈련된 모델
% 을 다시 만듭니다. 생성된 코드를 사용하여 동일한 모델을 새 데이터로 훈련시키는 것을 자동
% 화하거나, 모델을 프로그래밍 방식으로 훈련시키는 방법을 익힐 수 있습니다.
%
%  입력값:
%      trainingData: 앱으로 가져온 행렬과 동일한 개수의 열과 데이터형을 갖는 행렬입니
%       다.
%
%
%  출력값:
%      trainedModel: 훈련된 회귀 모델이 포함된 구조체입니다. 이 구조체에는 훈련된 모
%       델에 대한 정보가 포함된 다양한 필드가 들어 있습니다.
%
%      trainedModel.predictFcn: 새 데이터를 사용하여 예측하기 위한 함수입니다.
%
%      validationRMSE: 검증 RMSE를 나타내는 double형입니다. 검증 RMSE는 앱의 모델
%       창에 각 모델별로 표시됩니다.
%
% 새 데이터로 모델을 훈련시키려면 이 코드를 사용하십시오. 모델을 다시 훈련시키려면 명령줄
% 에서 원래 데이터나 새 데이터를 입력 인수 trainingData(으)로 사용하여 함수를 호출하십
% 시오.
%
% 예를 들어, 원래 데이터 세트 T(으)로 훈련된 회귀 모델을 다시 훈련시키려면 다음을 입력하
% 십시오.
%   [trainedModel, validationRMSE] = trainRegressionModel(T)
%
% 새 데이터 T2에서 반환된 'trainedModel'을(를) 사용하여 예측하려면 다음을 사용하십시
% 오.
%   yfit = trainedModel.predictFcn(T2)
%
% T2은(는) 훈련에 사용된 예측 변수 열만 포함하는 행렬이어야 합니다. 세부 정보를 보려면
% 다음을 입력하십시오.
%   trainedModel.HowToPredict

% MATLAB에서 2024-12-25 22:50:53에 자동 생성됨


% 예측 변수와 응답 변수 추출
% 이 코드는 모델을 훈련시키기에 적합한 형태로 데이터를
% 처리합니다.
% 입력값을 테이블로 변환
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4'});

predictorNames = {'column_3', 'column_4'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_2;
isCategoricalPredictor = [false, false];

% 회귀 모델 훈련
% 이 코드는 모든 모델 옵션을 지정하고 모델을 훈련시킵니다.
regressionNeuralNetwork = fitrnet(...
    predictors, ...
    response, ...
    'LayerSizes', [10 10], ...
    'Activations', 'relu', ...
    'Lambda', 0, ...
    'IterationLimit', 1000, ...
    'Standardize', true);

% 예측 함수를 사용하여 결과 구조체 생성
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
neuralNetworkPredictFcn = @(x) predict(regressionNeuralNetwork, x);
trainedModel.predictFcn = @(x) neuralNetworkPredictFcn(predictorExtractionFcn(x));

% 추가적인 필드를 결과 구조체에 추가
trainedModel.RegressionNeuralNetwork = regressionNeuralNetwork;
trainedModel.About = '이 구조체는 회귀 학습기 R2023a에서 내보낸 훈련된 모델입니다.';
trainedModel.HowToPredict = sprintf('새 예측 변수 열 행렬 X를 사용하여 예측하려면 다음을 사용하십시오. \n yfit = c.predictFcn(X) \n여기서 ''c''를 이 구조체를 나타내는 변수의 이름(예: ''trainedModel'')으로 바꾸십시오. \n \n이 모델은 2개의 예측 변수를 사용하여 훈련되었으므로 X는 정확히 2개의 열을 포함해야 합니다. \nX는 훈련 데이터와 정확히 동일한 순서와 형식의 예측 변수 열만 포함해야 합니다.\n 응답 변수 열이나 앱으로 가져오지 않은 열은 포함시키지 마십시오. \n \n자세한 내용은 <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>을(를) 참조하십시오.');

% 예측 변수와 응답 변수 추출
% 이 코드는 모델을 훈련시키기에 적합한 형태로 데이터를
% 처리합니다.
% 입력값을 테이블로 변환
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4'});

predictorNames = {'column_3', 'column_4'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_2;
isCategoricalPredictor = [false, false];

% 교차 검증 수행
partitionedModel = crossval(trainedModel.RegressionNeuralNetwork, 'KFold', 5);

% 검증 예측값 계산
validationPredictions = kfoldPredict(partitionedModel);

% 검증 RMSE 계산
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
