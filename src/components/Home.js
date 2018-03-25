import React from 'react';
import { Link } from 'react-router-dom';

import Layout from './Layout';

const Home = () => {
  return (
    <Layout>
      <p>Hello World of not pls ME</p>
      <p>
        <Link to="/dynamic">Navigate to DynamoPage </Link>
      </p>
    </Layout>
  );
};

export default Home;
