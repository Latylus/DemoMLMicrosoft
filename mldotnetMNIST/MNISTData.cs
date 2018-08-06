using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Runtime.Api;

namespace mldotnetMNIST
{
    //Représentation de nos données utilisées en entrée de notre modèle
    public class MNISTData
    {
        //L'etiquette de reference qui indique a quel chiffre correspond l image
        [Column(ordinal: "1", name: "Label")]
        public float Label; 
        
        //Une representation aplatie de l image 28x28 MNIST
        [VectorType(784), ColumnName("Features")]
        public float[] Features; 
    }
    //Représentation de nos données utilisées en tant que sortie de notre modèle
    public class MNISTPrediction
    {
        //Un tableau de score pour chaque chiffre, le score le plus eleve correspond au chiffre predit
        [ColumnName("Score")]
        public float[] PredictedLabels; 
    }
}
