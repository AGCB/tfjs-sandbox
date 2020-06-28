import React, {useState} from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import './App.css';

function App() {
  const [model, setModel] = useState(null);
  const [data, setData] = useState([]);









  async function getData() {
    const carsDataReq = await fetch("https://storage.googleapis.com/tfjs-tutorials/carsData.json");
    const carsData = await carsDataReq.json();
    const cleaned = carsData.map(car => ({
      mpg: car.Miles_per_Gallon,
      horsepower: car.Horsepower,
    }))
    // REMOVE DATASET MEMBERS WITHOUT OUR INTERESTED FIELDS
    .filter(car => car.mpg !== null && car.horsepower !== null)
    .map(datapoint => ({
      x:datapoint.horsepower,
      y:datapoint.mpg
    }));
    // SOURCE DATA NEEDS TO BE TURNED TO TENSORS.
    // ALSO WILL SHUFFLE AND NORMALIZE...

    await setData(cleaned);


    // DATA VISUALIZATION.
    const surface = {
      name: 'MPG vs HP',
      tab: 'charts'
    }
    tfvis.render.scatterplot(
      surface,
      {values: data},
      {
        xLabel:'horsepower!',
        yLabel:'milesPerGallon',
        height: 300,
    })
  }






  function toggleScatterPlot(e) {
    e.preventDefault();
    tfvis.visor().toggle()
  }

  function handleCreateModel(e) {
      e.preventDefault();
      function createModel() {

        // MAKE A MODEL
        //     .sequential() is a more straightforward API.
        //     inputs flow down to outputs...
        //     as opposed to multiple branches or inputs/outputs
        //     ... still this basic model is useful in many cases?
        //
        const model = tf.sequential();


        // ADD SINGLE HIDDEN LAYER
        // DENSE.. refers to...
        //     inputs will be multiplied by matrix(weights)
        // BIAS number is added to result
        //     this is already set to true by default so no need to include.
        // first layer needs INPUTSHAPE
        //     [1] refers to single dimension of Horsepower data.
        // UNITS refers to how big the matrix will be... 1 weight for each feature.
        //
        model.add(tf.layers.dense({inputShape: [1], units:1, useBias:true}))




        // ADD OUTPUT LAYER
        //     we want to output 1 number.
        //     for this example we don't really need an output layer since our
        //     only layer is the same shape... but we separate them anyway so that we
        //     have flexibility later. We could modify the number of units in the hidden layer
        model.add(tf.layers.dense({units:1}));

        return model;
      }
      const model = createModel();
      setModel(model)


      // MODEL VISUALIZATION
      //     be careful with modal vs model... frontEnd vs dataScience...
      //         I was stuck on this for a bit... typeoo
      //     NAME and TAB keys are for tfvis UI...
      //         what other keys do we have available?
      //
      tfvis.show.modelSummary({name: 'Our Model Summary.', tab: 'ModelSummary'}, model)
  }

  function convertToTensor(e) {
    e.preventDefault();
    let output = [...data];
    // console.log('here is our data', output);

    // goals of this function are..
    //     convert data to tensor.
    //     shuffle data. remember that normally the first item is 130/18
    return tf.tidy(() => {
      tf.util.shuffle(output);
      // console.log('did we just shuffle data??', output);
      // convert data to tensor.
      const inputs = output.map(d => d.x);
      const labels = output.map(d => d.y);
      // console.log('here are some inputs and labels', inputs, labels);

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      // normalize to range of 0-1 using min max scaling.
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();
      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(inputMax.sub(labelMin));

      setData({
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // return the min/max bounds so we can use them later
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      });
    })
  }

  async function handleTrainModel(e) {
    e.preventDefault();
    console.log('inside handleTrainModel()...');
    const {inputs, labels} = data;
    async function trainModel(model, inputs, labels) {
        // Prepare the model for training.
        model.compile({
          optimizer: tf.train.adam(),
          loss: tf.losses.meanSquaredError,
          metrics: ['mse'],
        });

        const batchSize = 32;
        const epochs = 50;

        return await model.fit(inputs, labels, {
          batchSize,
          epochs,
          shuffle: true,
          callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
          )
        });
    }

    await trainModel(model, inputs, labels);
    console.log('done training');
  }

  return (
    <div className="App">
      <h1>Making Predictions from 2d data</h1>
      <button onClick={getData}>GET DATA</button>
      <button onClick={e => toggleScatterPlot(e)}>TOGGLE SCATTERPLOT</button>
      <button onClick={e => handleCreateModel(e)}>createModel</button>
      <button onClick={e => convertToTensor(e)}>convert to tensor</button>
      <button onClick={e => handleTrainModel(e)}>trainModel</button>
      <div id="histogram-cont">

      </div>
  </div>
  );
}

export default App;
