import React, { Component } from "react";
import LinearProgress from '@material-ui/core/LinearProgress';
import { Box, Typography, TextField, Button, withStyles } from '@material-ui/core';
import UploadService from "../services/Upload";

const BorderLinearProgress = withStyles((theme) => ({
  root: {
    height: 15,
    borderRadius: 5,
  },
  colorPrimary: {
    backgroundColor: "#EEEEEE",
  },
  bar: {
    borderRadius: 5,
    backgroundColor: '#1a90ff',
  },
}))(LinearProgress);

export default class AppFields extends Component {
  constructor(props) {
    super(props);
    this.selectFile = this.selectFile.bind(this);
    this.upload = this.upload.bind(this);

    this.state = {
      currentFile: undefined,
      previewImage: undefined,
      progress: 0,
      imgtxt: "",
      imgtxt1: "",
      message: "",
      isError: false,
      processingTime: 0,
      processingTime1: 0,  // Add processingTime to state
    };
  }

  componentDidMount() {}

  selectFile(event) {
    this.setState({
      currentFile: event.target.files[0],
      previewImage: URL.createObjectURL(event.target.files[0]),
      progress: 0,
      imgtxt: "",
      imgtxt1: "",
      message: "",
      processingTime: 0,
      processingTime1: 0,  // Reset processingTime
    });
  }

  upload() {
    this.setState({
      progress: 0
    });

    UploadService.upload(this.state.currentFile, (event) => {
      this.setState({
        progress: Math.round((100 * event.loaded) / event.total),
        message: "Please wait, uploading and processing the image...",
      });
    })
      .then((response) => {
        console.log(response.data);  // Debug print
        this.setState({
          imgtxt: response.data.text,
          imgtxt1: response.data.text1,
          message: response.data.message,
          isError: false,
          processingTime: response.data.processing_time,
          processingTime1: response.data.processing_time1 // Update processingTime
        });
      })
      .catch((err) => {
        console.log(err);  // Debug print
        this.setState({
          progress: 0,
          message: "Timeout!, processing the image took more time",
          currentFile: undefined,
          imgtxt: "",
          imgtxt1: "",
          isError: true,
          processingTime: 0,
          processingTime1: 0   // Reset processingTime on error
        });
      });
  }

  render() {
    const {
      currentFile,
      previewImage,
      progress,
      message,
      imgtxt,
      imgtxt1,
      isError,
      processingTime,
      processingTime1  // Destructure processingTime from state
    } = this.state;

    return (
      <div className="container">
        <div className="mg20">
          <label htmlFor="btn-upload">
            <input
              id="btn-upload"
              name="image"
              style={{ display: 'none' }}
              type="file"
              accept="image/png, image/jpeg, image/gif, image/jpg"
              onChange={this.selectFile} />
            <Button
              className="btn-choose"
              variant="outlined"
              component="span">
              Import Image
            </Button>
          </label>
          <div className="file-name">
            {currentFile ? currentFile.name : null}
          </div>
          {currentFile && (
            <Box className="my20" display="flex" alignItems="center">
              <Box width="100%" mr={1}>
                <BorderLinearProgress variant="determinate" value={progress} />
              </Box>
              <Box minWidth={35}>
                <Typography variant="body2" color="textSecondary">{`${progress}%`}</Typography>
              </Box>
            </Box>
          )}
          <div className="out">
            {previewImage && (
              <div>
                <img className="preview my20" src={previewImage} alt="" />
              </div>
            )}

            {message && (
              <Typography variant="subtitle2" className={`upload-message ${isError ? "error" : ""}`}>
                {message}
              </Typography>
            )}
            <Button
                  className="btn-sq"
                  color="primary"
                  variant="contained"
                  component="span"
                  disabled={!currentFile}
                  onClick={this.upload}>
                  Start Predict
                </Button>
            <div className="output-container">
              <div className="output1">
                <Box mt={7}>
                  <TextField
                    id="outlined-textarea"
                    label="Text from image with parallelization"
                    className="outtxt"
                    value={imgtxt}
                    placeholder="Text"
                    multiline
                    variant="outlined"
                  />
                </Box>
                {processingTime > 0 && (
                  <Typography variant="subtitle2" className="processing-time">
                    Processing time: {processingTime.toFixed(5)}s
                  </Typography>
                )}
              </div>
              <div className="output2">
                <Box mt={7}>
                  <TextField
                    id="outlined-textarea1"
                    label="Text from image with sequence"
                    className="outtxt"
                    value={imgtxt1}
                    placeholder="Text"
                    multiline
                    variant="outlined"
                  />
                </Box>
                {processingTime1 > 0 && (
                  <Typography variant="subtitle2" className="processing-time">
                    Processing time: {processingTime1.toFixed(5)}s
                  </Typography>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }
}
