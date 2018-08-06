using System;
using System.Threading;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Text.RegularExpressions;
using Xunit;
using Xunit.Abstractions;

namespace mldotnetMNIST
{
    

   
    internal static class Program
    {


        static readonly string _dataPath = "train.csv";
        static readonly string _testDataPath = "test.csv";
        static readonly string _onnxPath = "mnist.onnx";
        static readonly string _modelPath = "mnist.zip";
        public static void Main(string[] args)
        {
            //On passe par une methode Main secondaire asynchrone pour pouvoir attendre la sauvegarde du modèle
            MainAsync(args).Wait(); 
        }

        static async Task MainAsync(string[] args)
        {

            var model = await TrainModel();
            //SaveToOnnx(model); Pas disponible en version 0.3 pour un modèle multiclasse
            Evaluate(model);

            Console.ReadLine();
        }


 
        public static async Task<PredictionModel<MNISTData, MNISTPrediction>> TrainModel()
        {
            var pipeline = new LearningPipeline();

            //Étape d'analyse du fichier situé au chemin _dataPath pour match avec le modèle MNISTData
            pipeline.Add(new TextLoader(_dataPath)
            {
                Arguments = new TextLoaderArguments
                {
                    Separator = new[] { ',' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoaderRange(0) },
                            Type = DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(1, 784) },
                            Type = DataKind.Num
                        }
                    }
                }
            });
           
            //Étape du classifieur qui nous donnera un score pour chaque chiffre possible
            pipeline.Add(new LogisticRegressionClassifier() { NormalizeFeatures = NormalizeOption.Yes }); //93.3% acc


            var model = pipeline.Train<MNISTData, MNISTPrediction>();

            //Sauvegarde du modèle sous fichier .zip
            await model.WriteAsync(_modelPath);

            Console.WriteLine("=============== End training ===============");
            Console.WriteLine("The model is saved to {0}", _modelPath);

            return model; 

        }


        public static void SaveToOnnx(PredictionModel model)
        {
            //Sauvegarde sous format ONNX (pas encore fonctionnel pour multiclass à la version 0.3)
            OnnxConverter converter = new OnnxConverter()
            {
                InputsToDrop = new[] { "Label" },
                OutputsToDrop = new[] { "Label", "Features" },
                Onnx = _onnxPath,
                Domain = "Onnx"
            };
            converter.Convert(model);
        }

        public static void PrintMatrix(ConfusionMatrix matrix)
        {
            //Simple fonction pour imprimer une matrice
            
            for(int i = 0; i < matrix.Order; i++)
            {
                string s = "";
                for (int j =0; j<matrix.Order;j++ )
                {
                    double val = matrix[i,j];
                    s+= " " + val;
                }
                Console.WriteLine(s);
            }
        }

        
        public static void Evaluate(PredictionModel<MNISTData, MNISTPrediction> model)
        {
            //Creation d'un analyseur de donnée pour le fichier test
            var testData = new TextLoader(_testDataPath)
            {
                Arguments = new TextLoaderArguments
                {
                    Separator = new[] { ',' },
                    HasHeader = true,
                    Column = new[]
                    {
                        new TextLoaderColumn()
                        {
                            Name = "Label",
                            Source = new [] { new TextLoaderRange(0) },
                            Type = DataKind.Num
                        },

                        new TextLoaderColumn()
                        {
                            Name = "Features",
                            Source = new [] { new TextLoaderRange(1, 784) },
                            Type = DataKind.Num
                        }
                    }
                }
            };

            var evaluator = new ClassificationEvaluator();

            //On évalue ici notre modèle selon les données test, les résultats statistiques sont contenus dans metrics
            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Console.WriteLine();
            Console.WriteLine("PredictionModel quality metrics evaluation");
            Console.WriteLine("------------------------------------------");

            //deux exemples de statistques utiles : la précision macroscopique et la matrice de confusion
            Console.WriteLine("Macro Acc : {0}", metrics.AccuracyMacro);
            Console.WriteLine("----------------------------------");
            Console.WriteLine("Confusion Matrix");
            PrintMatrix(metrics.ConfusionMatrix);



        }
    }
}
