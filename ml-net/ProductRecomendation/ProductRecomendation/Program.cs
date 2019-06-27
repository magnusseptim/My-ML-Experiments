using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace ProductRecomendation
{

    // Great thanks for dotnet/machinelearning-samples for sample that this code was based for : 
    // https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/MatrixFactorization_ProductRecommendation
    // Real data file avaiable from : https://snap.stanford.edu/data/amazon0312.html


    public class ProductEntry
    {
        [KeyType(count: 3200444)]
        public uint FromNodeId { get; set; }
        [KeyType(count: 3200444)]
        public uint ToNodeId { get; set; }
    }

    public class Score
    {
        public float Value { get; set; }
    }

    class Program
    {
        private static string BaseDataSetRelativePath = @"../../../Data";
        private static string TrainingDataRelativePath = $"{BaseDataSetRelativePath}/Amazon0312.txt";
        private static string TrainingDataLocation = GetAbsolutePath(TrainingDataRelativePath);

        static void Main(string[] args)
        {
            MLContext context = new MLContext();

            var trainData = context.Data.LoadFromTextFile(
                TrainingDataLocation,
                columns:
                    new[]
                    {
                        new TextLoader.Column("Label", DataKind.Single, 0),
                        new TextLoader.Column(name: nameof(ProductEntry.FromNodeId), dataKind: DataKind.UInt32, source: new [] { new TextLoader.Range(0) }, new KeyCount(3200444)),
                        new TextLoader.Column(name: nameof(ProductEntry.ToNodeId), dataKind: DataKind.UInt32, source: new [] { new TextLoader.Range(1) }, new KeyCount(3200444))
                    },
                hasHeader: true,
                separatorChar: '\t');

            var recomendation = context.Recommendation().Trainers.MatrixFactorization(CreateOptions());

            ITransformer model = recomendation.Fit(trainData);

            var engine = context.Model.CreatePredictionEngine<ProductEntry, Score>(model);

            PredictSamples(
                engine,
                GenerateRandomItems(1, 1000, 1000),
                GenerateRandomItems(1, 1000, 1000));

            Console.WriteLine("=============== End of process, hit any key to finish ===============");
            Console.ReadKey();
        }

        public static MatrixFactorizationTrainer.Options CreateOptions()
        {
            MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
            options.MatrixColumnIndexColumnName = nameof(ProductEntry.FromNodeId);
            options.MatrixRowIndexColumnName = nameof(ProductEntry.ToNodeId);
            options.LabelColumnName = "Label";
            options.LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass;
            options.Lambda = 0.025;
            return options;
        }

        public static string GetAbsolutePath(string relativeDatasetPath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativeDatasetPath);

            return fullPath;
        }

        public static IEnumerable<int> GenerateRandomItems(
            int start, 
            int max, 
            int size)
        {
            Random rand = new Random();
            return Enumerable.Range(0, size)
                                     .Select(i => new Tuple<int, int>(rand.Next(start, max), i))
                                     .OrderBy(i => i.Item1)
                                     .Select(i => i.Item2);
        }

        public static void PredictSamples(
            PredictionEngine<ProductEntry, Score> engine,
            IEnumerable<int> rand1,
            IEnumerable<int> rand2)
        {
            Score score;
            ProductEntry entry;

            for (int i = 0; i < rand1.Count(); i++)
            {
                entry = new ProductEntry()
                {
                    FromNodeId = (uint)rand1.ElementAt(i),
                    ToNodeId = (uint)rand2.ElementAt(i)
                };
                score = engine.Predict(entry);

                Print(entry, score);
            }
        }

        public static void Print(ProductEntry entry ,Score prediction)
        {
            if (prediction.Value > 0)
                Console.WriteLine($"\n For FromNodeId = {entry.FromNodeId} and  FromNodeId = {entry.ToNodeId} the predicted score is " + Math.Round(prediction.Value, 1));
        }
    }
}
