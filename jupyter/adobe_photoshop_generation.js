// Function to generate a random color
function getRandomColor() {
    var color = new SolidColor();
    color.rgb.red = Math.floor(Math.random() * 256); // 0 to 255
    color.rgb.green = Math.floor(Math.random() * 256); // 0 to 255
    color.rgb.blue = Math.floor(Math.random() * 256); // 0 to 255
    return color;
}

for (var i = 0; i < 100; i++) {
    // Create a new document
    app.documents.add(1024, 1024, 300, "New Document " + (i + 1), NewDocumentMode.RGB, DocumentFill.TRANSPARENT, 1, BitsPerChannelType.EIGHT);
    
    // Create a new art layer at the beginning of the current document
    var layerRef = app.activeDocument.artLayers.add();
    layerRef.name = "MyColorFillLayer";
    layerRef.blendMode = BlendMode.NORMAL;
    
    // Select all to apply a fill to the selection
    app.activeDocument.selection.selectAll();
    
    // Create a random color to be used with the fill command
    var colorRef = getRandomColor();
    
    // Now apply fill to the current selection
    app.activeDocument.selection.fill(colorRef);
    
    // Deselect to clean up selection borders
    app.activeDocument.selection.deselect();
    
    // Save the filled document as a JPEG in the specified folder with a unique name
    // var savePath = new File("D:/Repositories/CameraSourceRT/datasets/Photoshop_generative/MyColorFilledImage_" + (i + 1) + ".jpg");
    // var saveOptions = new JPEGSaveOptions();
    // saveOptions.quality = 12; // Maximum quality
    // app.activeDocument.saveAs(savePath, saveOptions, true);

    // Define the TIFF save options
    // var tiffSaveOptions = new TiffSaveOptions();
    // tiffSaveOptions.embedColorProfile = true;
    // tiffSaveOptions.alphaChannels = true;
    // tiffSaveOptions.layers = true;  // Preserve layers
    // tiffSaveOptions.imageCompression = TIFFEncoding.NONE; // No compression for maximum quality
    
    // Adjust the savePath to point to the TIFF file
    // var savePath = new File("D:/Repositories/CameraSourceRT/datasets/Photoshop_generative/MyColorFilledImage_" + (i + 1) + ".tif");

    // Save the filled document as a TIFF in the specified folder with a unique name
    // app.activeDocument.saveAs(savePath, tiffSaveOptions, true);

    // Define the PSD save options
    var psdSaveOptions = new PhotoshopSaveOptions();
    psdSaveOptions.embedColorProfile = true;
    psdSaveOptions.alphaChannels = true;
    psdSaveOptions.layers = true;  // Preserve layers
    
    // Adjust the savePath to point to the PSD file
    var savePath = new File("D:/Repositories/CameraSourceRT/datasets/Photoshop_generative/MyGeneratedImage_" + (i + 1) + ".psd");

    // Save the filled document as a PSD in the specified folder with a unique name
    app.activeDocument.saveAs(savePath, psdSaveOptions, true);
    
    // Close the document if you don't need it open anymore
    app.activeDocument.close(SaveOptions.DONOTSAVECHANGES);
}
