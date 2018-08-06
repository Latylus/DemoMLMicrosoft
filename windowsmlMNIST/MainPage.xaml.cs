﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using System.Runtime.InteropServices.WindowsRuntime;
using Windows.Foundation;
using Windows.Foundation.Collections;
using Windows.UI.Xaml;
using Windows.UI.Xaml.Controls;
using Windows.Storage;
using Windows.UI.Xaml.Media.Imaging;
using Windows.AI.MachineLearning.Preview;

namespace MNIST_Demo
{
    public sealed partial class MainPage : Page
    {
        private MNISTModel ModelGen = new MNISTModel();
        private MNISTModelInput ModelInput = new MNISTModelInput();
        private MNISTModelOutput ModelOutput = new MNISTModelOutput();
        private Helper helper = new Helper();

        RenderTargetBitmap renderBitmap = new RenderTargetBitmap();

        public MainPage()
        {
            this.InitializeComponent();
            
            // Set supported inking device types.
            inkCanvas.InkPresenter.InputDeviceTypes = Windows.UI.Core.CoreInputDeviceTypes.Mouse | Windows.UI.Core.CoreInputDeviceTypes.Pen | Windows.UI.Core.CoreInputDeviceTypes.Touch;
            inkCanvas.InkPresenter.UpdateDefaultDrawingAttributes(
                new Windows.UI.Input.Inking.InkDrawingAttributes()
                {
                    Color = Windows.UI.Colors.White,
                    Size = new Size(22, 22),
                    IgnorePressure = true,
                    IgnoreTilt = true,
                }
            );
            LoadModel();
        }

        private async void LoadModel()
        {
            //Chargement du modèle de machine learning
            StorageFile modelFile = await StorageFile.GetFileFromApplicationUriAsync(new Uri($"ms-appx:///Assets/MNIST.onnx"));
            ModelGen = await MNISTModel.CreateMNISTModel(modelFile);
        }

        private async void recognizeButton_Click(object sender, RoutedEventArgs e)
        {
            //On lie l'entrée du modèle avec le contenu du canevas que l'on a modifié pour correspondre au format attendu
            ModelInput.Input3 = await helper.GetHandWrittenImage(inkGrid);
            //Prédiction du modèle
            ModelOutput = await ModelGen.EvaluateAsync(ModelInput);

            //On trouve le chiffre avec la probabilité la plus élevée
            var maxIndex = ModelOutput.Plus214_Output_0.IndexOf(ModelOutput.Plus214_Output_0.Max());

            //On affiche ce chiffre
            numberLabel.Text = maxIndex.ToString();

        }

        private void clearButton_Click(object sender, RoutedEventArgs e)
        {
            inkCanvas.InkPresenter.StrokeContainer.Clear();
            numberLabel.Text = "";
        }
    }
}
