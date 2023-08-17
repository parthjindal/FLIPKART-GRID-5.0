var linkPreviewJs = require("link-preview-js");

// pass the link directly
linkPreviewJs
  .getLinkPreview("https://www.youtube.com/watch?v=MejbOFk7H6c")
  .then((data) => {
    console.debug(data);
  })
  .catch((error) => {
    console.error("An error occurred:", error);
  });