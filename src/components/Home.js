import React from 'react';
import { Link } from 'react-router-dom';
import Axios from 'axios';

import Layout from './Layout';

const Home = () => {
  var image = '=';
  function handleClick(e) {
    console.log("event")
    var config = {
      headers: {
        'Access-Control-Allow-Origin': '*',
        'Content-Type' : 'application/x-www-form-urlencoded'
      }
    };
    Axios.get('http://127.0.0.1:5000/v1/capture', config)
      .then(function(response){
          console.log(response)
          alert(response.data);
          console.log(response);
      }, function(error){
        alert("error");
        console.log(error);
      });
  }
  return (
    <Layout>
      <p>Hello World of not pls ME</p>
      <a href="#" onClick={handleClick}>
        Click me
      </a>
      <p>
        <Link to="/dynamic">Navigate to DynamoPage </Link>
      </p>

    </Layout>
  );
};

export default Home;
