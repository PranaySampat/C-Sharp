using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace RegressionAnalysis
{
        public class LogisticClassifier
    {
        private int numFeatures; // number of independent variables aka features
        private double[] weights; // b0 = constant
        private Random rnd;

        public LogisticClassifier(int numFeatures)
        {
            this.numFeatures = numFeatures; // number features/predictors
            this.weights = new double[numFeatures + 1]; // [0] = b0 constant
            this.rnd = new Random(0); // seed could be a parmeter to ctor
        }

        public double FindGoodL1Weight(double[][] trainData, int seed)
        {
            double result = 0.0;
            double bestErr = double.MaxValue;
            double currErr = double.MaxValue;
            double[] candidates = new double[] { 0.000, 0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150 };
            int maxEpochs = 1000;

            LogisticClassifier c = new LogisticClassifier(this.numFeatures);

            for (int trial = 0; trial < candidates.Length; ++trial)
            {
                double alpha1 = candidates[trial];
                double[] wts = c.TrainWeights(trainData, maxEpochs, seed, alpha1, 0.0);
                currErr = Error(trainData, wts, 0.0, 0.0);
                if (currErr < bestErr)
                {
                    bestErr = currErr;
                    result = candidates[trial];
                }
            }
            return result;
        }

        public double FindGoodL2Weight(double[][] trainData, int seed)
        {
            double result = 0.0;
            double bestErr = double.MaxValue;
            double currErr = double.MaxValue;
            double[] candidates = new double[] { 0.000, 0.001, 0.005, 0.010, 0.020, 0.050, 0.100, 0.150 };
            int maxEpochs = 1000;

            LogisticClassifier c = new LogisticClassifier(this.numFeatures);

            for (int trial = 0; trial < candidates.Length; ++trial)
            {
                double alpha2 = candidates[trial];
                double[] wts = c.TrainWeights(trainData, maxEpochs, seed, 0.0, alpha2);
                currErr = Error(trainData, wts, 0.0, 0.0);
                if (currErr < bestErr)
                {
                    bestErr = currErr;
                    result = candidates[trial];
                }
            }
            return result;
        }

        public double[] TrainWeights(double[][] trainData, int maxEpochs, int seed, double alpha1, double alpha2)
        {
            // use PSO. particle position == LR weights
            int numParticles = 10;
            double probDeath = 0.005;
            int dim = this.numFeatures + 1; // need one wt for each feature, plus the b0 constant

            Random rnd = new Random(seed);

            int epoch = 0;
            double minX = -10.0; // for each weight. assumes data has been normalized about 0
            double maxX = 10.0;
            double w = 0.729; // inertia weight
            double c1 = 1.49445; // cognitive/local weight
            double c2 = 1.49445; // social/global weight
            double r1, r2; // cognitive and social randomizations

            Particle[] swarm = new Particle[numParticles];
            // best solution found by any particle in the swarm. implicit initialization to all 0.0
            double[] bestSwarmPosition = new double[dim];
            double bestSwarmError = double.MaxValue; // smaller values better

            // initialize each Particle in the swarm with random positions and velocities
            for (int i = 0; i < swarm.Length; ++i)
            {
                double[] randomPosition = new double[dim];
                for (int j = 0; j < randomPosition.Length; ++j)
                    randomPosition[j] = (maxX - minX) * rnd.NextDouble() + minX;

                // randomPosition is a set of weights
                double error = Error(trainData, randomPosition, alpha1, alpha2);
                double[] randomVelocity = new double[dim];
                for (int j = 0; j < randomVelocity.Length; ++j)
                {
                    double lo = 0.1 * minX;
                    double hi = 0.1 * maxX;
                    randomVelocity[j] = (hi - lo) * rnd.NextDouble() + lo;
                }
                swarm[i] = new Particle(randomPosition, error, randomVelocity,
                    randomPosition, error); // last two are best-position and best-error

                // does current Particle have global best position?
                // best position for the particle is the one that's closest to the label (Y)
                if (swarm[i].error < bestSwarmError)
                {
                    bestSwarmError = swarm[i].error;
                    swarm[i].position.CopyTo(bestSwarmPosition, 0);
                }
            }

            // main PSO algorithm
            int[] sequence = new int[numParticles]; // process particles in random order
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            while (epoch < maxEpochs)
            {
                double[] newVelocity = new double[dim]; // step 1
                double[] newPosition = new double[dim]; // step 2
                double newError; // step 3

                Shuffle(sequence); // move particles in random sequence

                for (int pi = 0; pi < swarm.Length; ++pi) // each Particle (index)
                {
                    int i = sequence[pi];
                    Particle currP = swarm[i]; // for coding convenience

                    // 1. compute new velocity
                    for (int j = 0; j < currP.velocity.Length; ++j) // each x value of the velocity
                    {
                        r1 = rnd.NextDouble();
                        r2 = rnd.NextDouble();

                        // velocity depends on old velocity, best position of particle, and 
                        // best position of any particle
                        newVelocity[j] = (w * currP.velocity[j]) +
                            (c1 * r1 * (currP.bestPosition[j] - currP.position[j])) +
                            (c2 * r2 * (bestSwarmPosition[j] - currP.position[j]));
                    }

                    newVelocity.CopyTo(currP.velocity, 0);

                    // 2. use new velocity to compute new position
                    for (int j = 0; j < currP.position.Length; ++j)
                    {
                        newPosition[j] = currP.position[j] + newVelocity[j];  // compute new position
                        if (newPosition[j] < minX) // keep in range
                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }

                    newPosition.CopyTo(currP.position, 0);

                    // 3. use new position to compute new error
                    newError = Error(trainData, newPosition, alpha1, alpha2);
                    currP.error = newError;

                    if (newError < currP.bestError) // new particle best?
                    {
                        newPosition.CopyTo(currP.bestPosition, 0);
                        currP.bestError = newError;
                    }

                    if (newError < bestSwarmError) // new swarm best?
                    {
                        newPosition.CopyTo(bestSwarmPosition, 0);
                        bestSwarmError = newError;
                    }

                    // 4. optional: does curr particle die?
                    double die = rnd.NextDouble();
                    if (die < probDeath)
                    {
                        // new position, leave velocity, update error
                        for (int j = 0; j < currP.position.Length; ++j)
                            currP.position[j] = (maxX - minX) * rnd.NextDouble() + minX;
                        currP.error = Error(trainData, currP.position, alpha1, alpha2);
                        currP.position.CopyTo(currP.bestPosition, 0);
                        currP.bestError = currP.error;

                        if (currP.error < bestSwarmError) // swarm best by chance?
                        {
                            bestSwarmError = currP.error;
                            currP.position.CopyTo(bestSwarmPosition, 0);
                        }
                    }

                } // each Particle
                ++epoch;
            } // while

            double[] retResult = new double[dim];
            Array.Copy(bestSwarmPosition, retResult, retResult.Length);
            return retResult;
        } // TrainWeights

        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        public double Error(double[][] trainData, double[] weights, double alpha1, double alpha2)
        {
            // mean squared error using supplied weights
            // L1 regularization adds the sum of the absolute values of the weights
            // L2 regularization adds the sqrt of sum of squared values

            int yIndex = trainData[0].Length - 1; // y-value (0/1) is last column
            double sumSquaredError = 0.0;
            for (int i = 0; i < trainData.Length; ++i) // each data
            {
                double computed = ComputeY(trainData[i], weights);
                double desired = trainData[i][yIndex]; // ex: 0.0 or 1.0
                sumSquaredError += (computed - desired) * (computed - desired);
            }

            double sumAbsVals = 0.0; // L1 penalty
            for (int i = 0; i < weights.Length; ++i)
                sumAbsVals += Math.Abs(weights[i]);

            double sumSquaredVals = 0.0; // L2 penalty
            for (int i = 0; i < weights.Length; ++i)
                sumSquaredVals += (weights[i] * weights[i]);
            //double rootSum = Math.Sqrt(sumSquaredVals);

            return (sumSquaredError / trainData.Length) +
                (alpha1 * sumAbsVals) +
                (alpha2 * sumSquaredVals);
        }

        public double ComputeY(double[] dataItem, double[] weights)
        {
            double z = 0.0;

            z += weights[0]; // the b0 constant
            for (int i = 0; i < weights.Length - 1; ++i) // data might include Y
                z += (weights[i + 1] * dataItem[i]); // skip first weight

            return 1.0 / (1.0 + Math.Exp(-z));
        }

        public int ComputeDependent(double[] dataItem, double[] weights)
        {
            double sum = ComputeY(dataItem, weights);

            if (sum <= 0.5)
                return 0;
            else
                return 1;
        }

        public double Accuracy(double[][] trainData, double[] weights)
        {
            int numCorrect = 0;
            int numWrong = 0;
            int yIndex = trainData[0].Length - 1;
            for (int i = 0; i < trainData.Length; ++i)
            {
                double computed = ComputeDependent(trainData[i], weights); // implicit cast
                double desired = trainData[i][yIndex]; // 0.0 or 1.0

                double epsilon = 0.0000000001;
                if (Math.Abs(computed - desired) < epsilon)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numWrong + numCorrect);
        }

        public class Particle // for PSO training
        {
            public double[] position; // equivalent to weights
            public double error; // measure of fitness
            public double[] velocity; // determines new position
            public double[] bestPosition; // best found by this Particle
            public double bestError;

            public Particle(double[] position, double error, double[] velocity,
            double[] bestPosition, double bestError)
            {
                this.position = new double[position.Length];
                position.CopyTo(this.position, 0);
                this.error = error;
                this.velocity = new double[velocity.Length];
                velocity.CopyTo(this.velocity, 0);
                this.bestPosition = new double[bestPosition.Length];
                bestPosition.CopyTo(this.bestPosition, 0);
                this.bestError = bestError;
            }
        } // (nested) class Particle
    } // LogisticClassifier
    public class Program
{
        public static void Main(string[] args)
        {
               int numFeatures = 14;
            int numRows = 1000;
            int seed = 56;  // interesting seeds: 28, 32, (42), 56, 58, 63, 91

            // generate artificial observations
            Console.WriteLine("\nGenerating " + numRows +
                " artificial data items with " + numFeatures + " features");
            double[][] allData = MakeAllData(numFeatures, numRows, seed);
            

             // instantiate logistic binary classifier
            Console.WriteLine("Creating LR binary classifier..");
            LogisticClassifier lc = new LogisticClassifier(numFeatures);
                        // split into training and test datasets
            Console.WriteLine("Creating train (80%) and test (20%) matrices after scrambling observations order..");
            double[][] trainData;
            double[][] testData;
            MakeTrainTest(allData, 0, out trainData, out testData);
            Console.WriteLine("Done");
            Console.WriteLine("\nTraining data: \n");
            ShowMatrix(trainData, 4, 2, true);
            Console.WriteLine("\nTest data: \n");
            ShowMatrix(testData, 3, 2, true);
                        // train using no regularization
            int maxEpochs = 1000;
            Console.WriteLine("\nStarting training using no regularization..");
            double[] weights = lc.TrainWeights(trainData, maxEpochs, seed, 0.0, 0.0);

            Console.WriteLine("\nBest weights found:");
            ShowVector(weights, 3, weights.Length, true);

            double trainAccuracy = lc.Accuracy(trainData, weights);
            Console.WriteLine("Prediction accuracy on training data = " + trainAccuracy.ToString("F4"));

            double testAccuracy = lc.Accuracy(testData, weights);
            Console.WriteLine("Prediction accuracy on test data = " + testAccuracy.ToString("F4"));
                        // find L1 and L2
            Console.WriteLine("\nSeeking good L1 weight");
            //double alpha1 = lc.FindGoodL1Weight(trainData, seed);
            double alpha1 = 0.005;
            Console.WriteLine("Good L1 weight = " + alpha1.ToString("F3"));

            Console.WriteLine("\nSeeking good L2 weight");
            //double alpha2 = lc.FindGoodL2Weight(trainData, seed);
            double alpha2 = 0.001;
            Console.WriteLine("Good L2 weight = " + alpha2.ToString("F3"));
                        // train using L1 regularization
            Console.WriteLine("\nStarting training using L1 regularization, alpha1 = " + alpha1.ToString("F3"));
            weights = lc.TrainWeights(trainData, maxEpochs, seed, alpha1, 0.0);

            Console.WriteLine("\nBest weights found:");
            ShowVector(weights, 3, weights.Length, true);

            trainAccuracy = lc.Accuracy(trainData, weights);
            Console.WriteLine("Prediction accuracy on training data = " + trainAccuracy.ToString("F4"));

            testAccuracy = lc.Accuracy(testData, weights);
            Console.WriteLine("Prediction accuracy on test data = " + testAccuracy.ToString("F4"));
                        // train using L2 regularization
            Console.WriteLine("\nStarting training using L2 regularization, alpha2 = " + alpha2.ToString("F3"));
            weights = lc.TrainWeights(trainData, maxEpochs, seed, 0.0, alpha2);

            Console.WriteLine("\nBest weights found:");
            ShowVector(weights, 3, weights.Length, true);

            trainAccuracy = lc.Accuracy(trainData, weights);
            Console.WriteLine("Prediction accuracy on training data = " + trainAccuracy.ToString("F4"));

            testAccuracy = lc.Accuracy(testData, weights);
            Console.WriteLine("Prediction accuracy on test data = " + testAccuracy.ToString("F4"));

            Console.WriteLine("\nEnd Regularization demo\n");
            Console.ReadLine();
        } // Main Method
                // generate artificial observations
        static double[][] MakeAllData(int numFeatures, int numRows, int seed)
        {
            Random rnd = new Random(seed);

            // numfeatures weights (bills)
            double[] weights = new double[numFeatures + 1]; // inc. b0
            for (int i = 0; i < weights.Length; ++i)
                weights[i] = 20.0 * rnd.NextDouble() - 10.0; // [-10.0 to +10.0]

            // numRows observations (congressmen voting on each bill) 
            // Last column reserved for label, which is categorical binary
            double[][] data = new double[numRows][]; // allocate matrix
            for (int i = 0; i < numRows; ++i)
                data[i] = new double[numFeatures + 1]; // Y in last column
            
            data[0] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[1] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[2] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[3] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[4] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[5] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[6] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[7] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[8] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[9] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[10] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[11] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[12] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[13] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[14] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[15] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[16] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[17] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[18] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[19] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[20] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[21] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[22] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[23] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[24] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[25] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[26] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[27] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[28] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[29] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[30] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[31] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[32] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[33] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[34] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[35] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[36] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[37] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[38] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[39] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[40] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[41] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[42] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[43] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[44] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[45] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[46] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[47] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[48] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[49] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[50] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[51] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[52] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[53] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[54] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[55] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[56] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[57] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[58] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[59] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[60] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[61] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[62] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[63] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[64] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[65] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[66] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[67] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[68] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[69] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[70] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[71] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[72] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[73] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[74] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[75] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[76] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[77] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[78] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[79] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[80] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[81] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[82] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[83] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[84] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[85] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[86] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[87] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[88] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[89] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[90] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00};
data[91] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[92] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[93] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00};
data[94] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[95] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[96] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[97] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[98] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[99] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[100] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[101] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[102] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[103] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[104] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[105] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[106] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[107] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[108] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[109] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[110] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[111] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[112] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[113] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[114] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[115] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[116] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[117] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[118] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[119] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[120] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[121] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[122] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[123] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[124] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[125] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[126] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[127] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[128] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[129] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[130] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[131] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[132] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[133] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[134] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[135] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[136] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[137] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[138] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[139] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[140] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[141] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[142] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[143] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[144] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[145] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[146] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[147] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[148] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[149] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[150] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[151] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[152] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[153] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[154] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[155] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[156] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[157] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[158] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[159] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[160] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[161] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[162] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[163] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[164] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[165] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[166] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[167] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[168] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[169] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[170] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[171] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[172] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[173] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[174] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[175] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[176] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[177] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[178] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[179] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[180] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[181] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[182] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[183] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[184] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[185] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[186] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[187] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[188] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[189] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00};
data[190] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[191] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[192] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00};
data[193] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[194] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[195] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[196] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[197] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[198] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[199] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[200] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[201] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[202] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[203] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[204] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[205] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[206] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[207] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00};
data[208] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[209] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[210] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00};
data[211] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[212] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[213] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[214] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[215] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[216] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[217] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[218] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[219] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[220] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[221] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[222] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[223] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[224] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[225] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[226] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[227] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[228] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[229] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[230] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[231] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[232] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[233] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[234] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[235] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[236] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[237] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[238] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[239] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[240] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[241] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[242] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[243] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[244] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[245] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[246] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[247] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[248] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[249] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[250] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[251] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[252] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[253] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[254] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[255] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[256] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[257] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[258] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[259] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[260] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[261] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[262] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[263] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[264] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[265] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[266] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[267] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[268] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[269] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[270] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[271] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[272] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[273] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[274] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[275] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[276] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[277] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[278] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[279] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[280] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[281] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[282] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[283] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[284] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[285] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[286] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[287] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[288] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[289] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[290] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[291] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[292] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[293] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[294] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[295] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[296] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[297] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[298] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[299] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[300] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[301] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[302] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[303] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[304] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[305] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[306] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[307] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[308] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[309] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[310] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[311] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[312] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[313] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[314] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[315] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[316] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[317] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[318] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[319] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[320] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[321] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[322] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[323] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[324] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[325] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[326] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[327] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[328] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[329] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[330] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[331] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[332] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[333] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[334] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[335] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[336] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[337] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[338] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[339] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[340] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[341] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[342] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[343] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[344] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[345] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[346] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[347] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[348] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[349] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[350] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[351] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[352] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[353] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[354] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[355] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[356] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[357] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[358] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[359] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[360] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[361] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[362] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[363] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[364] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[365] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[366] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[367] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[368] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[369] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[370] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[371] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[372] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[373] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[374] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[375] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[376] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[377] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[378] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[379] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[380] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[381] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[382] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[383] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[384] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[385] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[386] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[387] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[388] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[389] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[390] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[391] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[392] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[393] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[394] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[395] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[396] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[397] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[398] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[399] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[400] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[401] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[402] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[403] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[404] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[405] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[406] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[407] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[408] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[409] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[410] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[411] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[412] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[413] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[414] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[415] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[416] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[417] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[418] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[419] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[420] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[421] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[422] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[423] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[424] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[425] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[426] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[427] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[428] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[429] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[430] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[431] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[432] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[433] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[434] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[435] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[436] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[437] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[438] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[439] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[440] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[441] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[442] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[443] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[444] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[445] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[446] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[447] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[448] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[449] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[450] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[451] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[452] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[453] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[454] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[455] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[456] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[457] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[458] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[459] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[460] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[461] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[462] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[463] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[464] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[465] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[466] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[467] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[468] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[469] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[470] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[471] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[472] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[473] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[474] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[475] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[476] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[477] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[478] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[479] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[480] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[481] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[482] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[483] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[484] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[485] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[486] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[487] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[488] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[489] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[490] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[491] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[492] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[493] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[494] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[495] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[496] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[497] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[498] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[499] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[500] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[501] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[502] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[503] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[504] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[505] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[506] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[507] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[508] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[509] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[510] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[511] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[512] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[513] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[514] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[515] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[516] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[517] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[518] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[519] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[520] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[521] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[522] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[523] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[524] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[525] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[526] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[527] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[528] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[529] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[530] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[531] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[532] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[533] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[534] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[535] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[536] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[537] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[538] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[539] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[540] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[541] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[542] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[543] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[544] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[545] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[546] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[547] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[548] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[549] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[550] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[551] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[552] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[553] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[554] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[555] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[556] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[557] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[558] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[559] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[560] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[561] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[562] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[563] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[564] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[565] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[566] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[567] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[568] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[569] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[570] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[571] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[572] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[573] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[574] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[575] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[576] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[577] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[578] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[579] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[580] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[581] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[582] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[583] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[584] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[585] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[586] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[587] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[588] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[589] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[590] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[591] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[592] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[593] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[594] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[595] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[596] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[597] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[598] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[599] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[600] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[601] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[602] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[603] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[604] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[605] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[606] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[607] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[608] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[609] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[610] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[611] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[612] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[613] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[614] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[615] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[616] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[617] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[618] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[619] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[620] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[621] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[622] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[623] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[624] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[625] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[626] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[627] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[628] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[629] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[630] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[631] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[632] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[633] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[634] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[635] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[636] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[637] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[638] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[639] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[640] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[641] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[642] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[643] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[644] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[645] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[646] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[647] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[648] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[649] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[650] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[651] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[652] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[653] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[654] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[655] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[656] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[657] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[658] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[659] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[660] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[661] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[662] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[663] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[664] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[665] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[666] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[667] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[668] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[669] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[670] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[671] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[672] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[673] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[674] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[675] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[676] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[677] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[678] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[679] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[680] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[681] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[682] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[683] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[684] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[685] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[686] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[687] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[688] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[689] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[690] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[691] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[692] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[693] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[694] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[695] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[696] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[697] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[698] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[699] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[700] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[701] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[702] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[703] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[704] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[705] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[706] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[707] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[708] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[709] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[710] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[711] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[712] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[713] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[714] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[715] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[716] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[717] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[718] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[719] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[720] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[721] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[722] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[723] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[724] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[725] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[726] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[727] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[728] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[729] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[730] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[731] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[732] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[733] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[734] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[735] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[736] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[737] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[738] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[739] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[740] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[741] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[742] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[743] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[744] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[745] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[746] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[747] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[748] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[749] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[750] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[751] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[752] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[753] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[754] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[755] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[756] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[757] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[758] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[759] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[760] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[761] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[762] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[763] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[764] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[765] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[766] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[767] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[768] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[769] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[770] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[771] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[772] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[773] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[774] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[775] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[776] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[777] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[778] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[779] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[780] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[781] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[782] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[783] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[784] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[785] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[786] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[787] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[788] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[789] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[790] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[791] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[792] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[793] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[794] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[795] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[796] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[797] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[798] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[799] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[800] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[801] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[802] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[803] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[804] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[805] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[806] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[807] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[808] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[809] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[810] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[811] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[812] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[813] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[814] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[815] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[816] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[817] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[818] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[819] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[820] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[821] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[822] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[823] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[824] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[825] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[826] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[827] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[828] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[829] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[830] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[831] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[832] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[833] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[834] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[835] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[836] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[837] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[838] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[839] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[840] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[841] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[842] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[843] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[844] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[845] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[846] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[847] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[848] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[849] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[850] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[851] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[852] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[853] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[854] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[855] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[856] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[857] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[858] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[859] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[860] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[861] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[862] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[863] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[864] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[865] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[866] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[867] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[868] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[869] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[870] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[871] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[872] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[873] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[874] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[875] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[876] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[877] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[878] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[879] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[880] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[881] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[882] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[883] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[884] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[885] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[886] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[887] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[888] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[889] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[890] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[891] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[892] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[893] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[894] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[895] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[896] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[897] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[898] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[899] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[900] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[901] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[902] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[903] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[904] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[905] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[906] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[907] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[908] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[909] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[910] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[911] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[912] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[913] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[914] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[915] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[916] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[917] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[918] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[919] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[920] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[921] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[922] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[923] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[924] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[925] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[926] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[927] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[928] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[929] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[930] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[931] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[932] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[933] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[934] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[935] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[936] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[937] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[938] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[939] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[940] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[941] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[942] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[943] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[944] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[945] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[946] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[947] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[948] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[949] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[950] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[951] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[952] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[953] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[954] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[955] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[956] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[957] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[958] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[959] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[960] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[961] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[962] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[963] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[964] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[965] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[966] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[967] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[968] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[969] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[970] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[971] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[972] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[973] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[974] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[975] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[976] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[977] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[978] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[979] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[980] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[981] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[982] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[983] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[984] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[985] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[986] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[987] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[988] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00 ,0.00};
data[989] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00};
data[990] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[991] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[992] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[993] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[994] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,0.00 ,0.00};
data[995] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,1.00 ,1.00};
data[996] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[997] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,1.00 ,0.00 ,0.00};
data[998] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};
data[999] = new double[] {0.00 ,1.00 ,0.00 ,1.00 ,0.00 ,0.00 ,1.00 ,1.00 ,1.00 ,1.00 ,0.00 ,0.00 ,0.00 ,1.00 ,0.00};

            //generate random observations
            // for (int i = 0; i < numRows; ++i) // for each row
            // {
            //     double y = weights[0]; // the b0 
            //     for (int j = 0; j < numFeatures; ++j) // each feature / column except last
            //     {
            //         double x = 20.0 * rnd.NextDouble() - 10.0; // random X in [10.0, +10.0]
            //         result[i][j] = x; // store x
                    
            //         double wx = x * weights[j + 1]; // weight * x 
            //         y += wx; // accumulate to get Y
            //         // now add some noise
            //         y += numFeatures * rnd.NextDouble();
            //     }
            //     if (y > numFeatures) // because the noise was +, make it harder to be a 1
            //         result[i][numFeatures] = 1.0; // store y in last column
            //     else
            //         result[i][numFeatures] = 0.0;
            // }
            
            Console.WriteLine("Data generation weights:");
            ShowVector(weights, 4, 10, true);

            Console.WriteLine("\nFirst few lines of all data are (last column is the label): \n");
            ShowMatrix(data, 4, 4, true);

            return data;
        }
                static void MakeTrainTest(double[][] allData, int seed, out double[][] trainData, out double[][] testData)
        {
            Random rnd = new Random(seed);
            int totRows = allData.Length;
            int numTrainRows = (int)(totRows * 0.80); // 80% hard-coded
            int numTestRows = totRows - numTrainRows;
            trainData = new double[numTrainRows][];
            testData = new double[numTestRows][];

            double[][] copy = new double[allData.Length][]; // ref copy of all data
            for (int i = 0; i < copy.Length; ++i)
                copy[i] = allData[i];

            for (int i = 0; i < copy.Length; ++i) // scramble order
            {
                int r = rnd.Next(i, copy.Length); // use Fisher-Yates
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }
            for (int i = 0; i < numTrainRows; ++i)
                trainData[i] = copy[i];

            for (int i = 0; i < numTestRows; ++i)
                testData[i] = copy[i + numTrainRows];
        } //end of Method
                public static void ShowVector(double[] vector, int decimals, int lineLen, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % lineLen == 0) Console.WriteLine("");
                if (vector[i] >= 0) Console.Write(" ");
                Console.Write(vector[i].ToString("F" + decimals) + " ");
            }
            if (newLine == true)
                Console.WriteLine("");
        }

        static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool indices)
        {
            for (int i = 0; i < numRows; ++i)
            {
                if (indices == true)
                    Console.Write("[" + i.ToString().PadLeft(2) + "]   ");
                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    Console.Write(matrix[i][j].ToString("F" + decimals) + " ");
                }
                Console.WriteLine("");
            }
            int lastIndex = matrix.Length - 1;
            if (indices == true)
                Console.Write("[" + lastIndex.ToString().PadLeft(2) + "]   ");
            for (int j = 0; j < matrix[lastIndex].Length; ++j)
                Console.Write(matrix[lastIndex][j].ToString("F" + decimals) + " ");
            Console.WriteLine("\n");
        }
    }
}
