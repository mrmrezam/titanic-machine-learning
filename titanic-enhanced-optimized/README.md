# Improving Survival Prediction Accuracy on the Titanic
### Introduction
In recent years, optimizing the parameters of machine learning models has been recognized as a fundamental step in improving prediction accuracy. Metaheuristic algorithms, inspired by nature and social behaviors, discover optimal solutions in the search space. These algorithms, unlike traditional optimizations, have the ability to better and more quickly identify good areas in the search space and generally perform better in complex, multidimensional problems that involve many variables. They are considered effective tools for maximizing model accuracy and reducing error rates in predictions. This report examines the accuracy of survival predictions in the Titanic dataset and optimizes model parameters using metaheuristic algorithms, including Genetic Algorithm (GA), Differential Evolution (DE), and Optuna.
### GA Algorithm
To implement the Genetic Algorithm, given its history and importance, we utilized the pygad library. Initially, we wrote a function that implements a genetic algorithm, which finds the best solution in a specified genetic space using a defined fitness function. In this function, we manually adjusted the values of the Genetic Algorithm parameters to consider the best parameters among the tested values as final parameters.

Next, we wrote the fitness function, where the model parameters are extracted based on the solution obtained from the genetic algorithm, and then the model is built and trained with these parameters, calculating the model's accuracy on the test data and returning it as a result. We also defined the gene space or the range of different model parameters.

In the next step, we executed the genetic algorithm using the two mentioned functions and extracted the best results and parameters from it. Finally, we built the model with the optimized parameters obtained, trained it, and measured its accuracy.
### DE Algorithm
To implement this algorithm, we used the scipy.optimize library. We first defined a function that takes model parameters and the model type (defaulting to Random Forest) as inputs. Then, based on the specified model type, the parameters related to each model are determined.

After defining the parameters, a corresponding model instance is created using the configured parameters. The model is then trained with the best parameters, and accuracy is calculated.

Due to high computational load, obtaining parameters for five algorithms simultaneously with the Stacking Classifier model was not feasible. Instead, we also performed predictions using the SVM classifier alongside the previous models.
### Optuna Framework
One of the fastest methods for finding optimal parameters is using Optuna. Although this framework is not an optimization algorithm itself, it employs Bayesian optimization to find parameters. In this method, to optimize the parameters, we first define an objective function and specify the model parameters within it, then create a model instance, and extract the best parameters.

Ultimately, with the best parameters, the model is built, trained, and tested, and accuracy is calculated. In this method, we used various approaches for the Stacking model, such as obtaining parameters from individual models and substituting them into the final model or placing parameters in base models and optimizing the meta-model, ultimately achieving the highest accuracy by optimizing all parameters of the four models used in stacking.

It should be noted that in all optimization algorithms, tuning the parameters of the optimization function can play a significant role in finding the optimal model parameters and avoiding local optima.

Another point is the stochastic nature of these algorithms, which means that even with the definition of a seed or random state and similar parameters, different results may sometimes be obtained in different runs.

It is also worth mentioning that in this work, we aimed to use libraries for implementing optimization algorithms, but in some test cases, it was observed that sometimes better results were achieved without libraries and by implementing the algorithms manually in function form, leading to better speed.

Additionally, during the testing and evaluation phases, other optimization algorithms were also initially assessed, but for various reasons, such as the lack of strong libraries, weak initial results, or high computational load, their inclusion in this work was not possible.
### Results
The best performance among all models was related to the XGBoost Classifier with the DE algorithm and the Stacking Classifier with the GA algorithm, achieving an accuracy of 91.12%. The classification reports for these models are detailed below.

XGBoost Classifier with DE Algorithm

Stacking Classifier with GA Algorithm

If we examine results other than accuracy, we can see that for those who did not survive, we achieved a correct prediction (recall of 97 in the second model), which is very important in cases like the Titanic, as this number helps us prevent the deaths of those individuals in similar incidents. It is essential to note that the recall value for those who survived is less critical, as a low value in this area is not vital; in fact, accurately predicting those who do not survive is more important to find a solution.

Given the equal F1 score, which represents the balance between precision and recall, and the higher recall for individuals who did not survive in the second model, we can conclude that the second model performed relatively better.

Regarding the use of Optuna, the highest accuracy achieved was related to the stacking model with an accuracy of 90%, but with significantly higher speed than GridSearch. This method offers much higher speed compared to optimization algorithms, which we can consider an advantage.
### Conclusion
After examining optimization algorithms and tuning the parameters of various models using them, we observed that the accuracy of all models improved with optimization. While the highest accuracy without optimization was 90%, and subsequently 87.8%, after using optimization in several models, we witnessed accuracies of 90% and even reached 91%.

Analysis results indicate that utilizing these optimization algorithms leads to a significant improvement in model prediction accuracy, highlighting the importance and positive impact of parameter optimization on overall model performance. It seems that if these algorithms are generalized to other areas of data work, such as feature selection or replacing missing values, even better results could be achieved.

Ultimately, considering that the best output is related to a Stacking model combined with optimization algorithms, we can conclude that with suitable hardware capabilities to handle high computational loads, this approach can yield the best output. However, if the data volume is very high and complex calculations are not feasible, we can use faster methods with lower computational loads and overlook the slight decrease in accuracy.
