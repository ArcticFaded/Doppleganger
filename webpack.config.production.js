const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const ExtractTextPlugin = require('extract-text-webpack-plugin');

module.exports = {
  mode: 'production',
  entry: {
    vendor: ['semantic-ui-react'],
    app: './src/index.js'
  },
  output: {
    // We want to create the JavaScript bundles under a
    // 'static' directory
    filename: 'static/[name].[hash].js',

    // Absolute path to the desired output directory. In our
    //case a directory named 'dist'
    // '__dirname' is a Node variable that gives us the absolute
    // path to our current directory. Then with 'path.resolve' we
    // join directories
    // Webpack 4 assumes your output path will be './dist' so you
    // can just leave this
    // entry out.
    path: path.resolve(__dirname, 'dist'),

    publicPath: '/'
  },

  // Change to production source maps
  devtool: 'source-map',

  module: {
    rules: [
      {
        test: /\.(js)$/,
        exclude: /node_modules/,
        use: ['babel-loader']
      },
      {
        test: /\.css$/,

        // We configure 'Extract Text Plugin'
        use: ExtractTextPlugin.extract({

          // loader that should be used when the
          // CSS is not extracted
          fallback: 'style-loader',

          use: [
            {
              loader: 'css-loader',
              options: {
                modules: true,

                // Allows to configure how many loaders
                // before css-loader should be applied
                // to @import(ed) resources
                importLoaders: 1,

                camelCase: true,

                // Create source maps for CSS files
                sourceMap: true
              }
            },
            {
              // PostCSS will run before css-loader and will
              // minify and autoprefix our CSS rules. We are also
              // telling it to only use the last 2
              // versions of the browsers when autoprefixing
              loader: 'postcss-loader',
              options: {
                config: {
                  ctx: {
                    autoprefixer: {
                      browsers: 'last 2 versions'
                    }
                  }
                }
              }
            }
          ]
        })
      }
    ]
  },
  optimization: {
    splitChunks: {
      cacheGroups: {
        vendor: {
          chunks: 'initial',
          test: 'vendor',
          name: 'vendor',
          enforce: true
        }
      }
    }
  },
  plugins: [
    new HtmlWebpackPlugin({
      template: 'public/index.html',
      favicon: 'public/favicon.ico'
    }),

    // Create the stylesheet under 'styles' directory
    new ExtractTextPlugin({
      filename: 'styles/styles.[contenthash].css',
      allChunks: true
    })
  ]
};
