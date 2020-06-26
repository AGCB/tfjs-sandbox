import React, {useState} from 'react';
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import './App.css';

function App() {
  const [myTog, setMyTog] = useState([false, true, 3, 'foo']); // default comp state
  const [data, setData] = useState([]);

  // get Car Dataset..
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
    // SOURCE DATA IS IN GOOD ORDER NOW
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
      // MODEL VISUALIZATION
      //     be careful with modal vs model... frontEnd vs dataScience...
      //         I was stuck on this for a bit... typeoo
      //     NAME and TAB keys are for tfvis UI...
      //         what other keys do we have available?
      //
      tfvis.show.modelSummary({name: 'Our Model Summary.', tab: 'ModelSummary'}, model)
  }


  return (
    <div className="App">
      <h1>Making Predictions from 2d data</h1>
      <button onClick={getData}>GET DATA</button>
      <button onClick={e => toggleScatterPlot(e)}>TOGGLE SCATTERPLOT</button>
      <button onClick={e => handleCreateModel(e)}>createModel</button>

      <div id="histogram-cont">

      </div>
  </div>
  );
}

export default App;
