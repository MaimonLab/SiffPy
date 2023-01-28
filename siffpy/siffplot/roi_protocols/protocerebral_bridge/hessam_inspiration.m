function varargout = roiDetectGUI(varargin)
    % ROIDETECTGUI MATLAB code for roiDetectGUI.fig
    %      ROIDETECTGUI, by itself, creates a new ROIDETECTGUI or raises the existing
    %      singleton*.
    %
    %      H = ROIDETECTGUI returns the handle to a new ROIDETECTGUI or the handle to
    %      the existing singleton*.
    %
    %      ROIDETECTGUI('CALLBACK',hObject,eventData,handles,...) calls the local
    %      function named CALLBACK in ROIDETECTGUI.M with the given input arguments.
    %
    %      ROIDETECTGUI('Property','Value',...) creates a new ROIDETECTGUI or raises the
    %      existing singleton*.  Starting from the left, property value pairs are
    %      applied to the GUI before roiDetectGUI_OpeningFcn gets called.  An
    %      unrecognized property name or invalid value makes property application
    %      stop.  All inputs are passed to roiDetectGUI_OpeningFcn via varargin.
    %
    %      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
    %      instance to run (singleton)".
    %
    % See also: GUIDE, GUIDATA, GUIHANDLES
    
    % Edit the above text to modify the response to help roiDetectGUI
    
    % Last Modified by GUIDE v2.5 05-Aug-2022 11:01:14
    
    % Begin initialization code - DO NOT EDIT
    gui_Singleton = 1;
    gui_State = struct('gui_Name',       mfilename, ...
                       'gui_Singleton',  gui_Singleton, ...
                       'gui_OpeningFcn', @roiDetectGUI_OpeningFcn, ...
                       'gui_OutputFcn',  @roiDetectGUI_OutputFcn, ...
                       'gui_LayoutFcn',  [] , ...
                       'gui_Callback',   []);
    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end
    
    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end
    % End initialization code - DO NOT EDIT
    
    
    % --- Executes just before roiDetectGUI is made visible.
    function roiDetectGUI_OpeningFcn(hObject, eventdata, handles, varargin)
    % This function has no output args, see OutputFcn.
    % hObject    handle to figure
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    % varargin   command line arguments to roiDetectGUI (see VARARGIN)
    
    % Choose default command line output for roiDetectGUI
    handles.output = hObject;
    
    hObject.UserData.ROIs = [];
    hObject.UserData.tuningVectors = []; % an array of complex numbers
    hObject.UserData.identityTracker = []; % maintains a unique number for each ROI that can be matched with activity profile and other data
    hObject.UserData.maxIdentityNum = 0;
    
    hObject.UserData.activityProfiles = {};
    
    hObject.UserData.history = [];
    hObject.UserData.history(1).ROIs = hObject.UserData.ROIs;
    hObject.UserData.history(1).tuningVectors = hObject.UserData.tuningVectors;
    hObject.UserData.history(1).identityTracker = hObject.UserData.identityTracker;
    hObject.UserData.historyIndex = 1;
    
    handles.listboxROIs.String = {};
    
    flatten = @(x) x(:);
    stackZAlongYKeepT = @(M) permute(reshape(permute(M, [2 1 4 3]), [size(M, 2), size(M, 1)*size(M, 4), size(M, 3)]), [2 1 3 4]);
    
    if(ischar(varargin{1}))
        assert(strcmp(varargin{1}(end-3:end), '.mat'));
        fprintf('Loading imageStack from %s...', varargin{1});
        load(varargin{1}, 'imageStack', 'timestamps');
        fprintf('[DONE]\n');
        if(size(imageStack, 3) < size(imageStack, 4))
            answer = questdlg(sprintf('It seems like the number of slices (%d) and numeber of time frames (%d) are stored in reverse. Would you like to correct that?', size(imageStack, 3), size(imageStack, 4)), ...
                'Dimension Mismatch', ...
                'Yes (Overwrite corrected version)', 'Yes (Don''t modify file)', 'No, it'' correct', 'Yes (Save corrected version)');
            switch answer
                case 'Yes (Overwrite corrected version)'
                    imageStack = permute(imageStack, [1 2 4 3]);
                    fprintf('Saving...');
                    save(varargin{1}, 'imageStack', '-append');
                    fprintf('[DONE]\n');
                case 'Yes (Don''t modify file)'
                    imageStack = permute(imageStack, [1 2 4 3]);
            end
        end
        
        hObject.UserData.imageStack = stackZAlongYKeepT(imageStack);
        hObject.UserData.originalDims = size(imageStack);
        clear imageStack;
        
        if(exist('timestamps', 'var'))
            assert(size(timestamps, 3) == hObject.UserData.originalDims(3) && size(timestamps, 4) == hObject.UserData.originalDims(4));
            flatten = @(x) x(:);
            hObject.UserData.timestamps = flatten(mean(timestamps, 4));
        else
            hObject.UserData.timestamps = [];
            warning('timestamps not found for %s', varargin{1});
        end
        
        if(contains(varargin{1}, 'registered'))
            hObject.UserData.fileName = varargin{1};
        end
        
        handles.mainFig.Name = [handles.mainFig.Name, ': ', varargin{1}];
    else
        hObject.UserData.imageStack = stackZAlongYKeepT(varargin{1});
        hObject.UserData.originalDims = size(varargin{1});
    end
    
    handles = initialize_video_axes(handles);
    assert(isfield(handles, 'thresholdingContourHandle'));
    
    set(hObject, 'toolbar', 'figure');
    set(hObject, 'menubar', 'figure');
    
    T = hObject.UserData.originalDims(3);
    handles.avgWidthSlider.base = 1.1; % with every tick the width is multiplied by this value
    b = handles.avgWidthSlider.base;
    set(handles.sliderAvgWinWidth, 'max', log(T)/log(b), 'min', 0, 'sliderstep', [log(b)/log(T) 0.1], 'Value', 1);
    w = round(b^handles.sliderAvgWinWidth.Value);
    set(handles.sliderTimeAxis, 'max', T-w+1, 'min', 1, 'sliderstep', [1 w]/(T-w), 'Value', 1);
    set(handles.sliderImageGain, 'max', 2, 'min', -0.5, 'sliderstep', [0.01 0.1], 'Value', 0);
    set(handles.sliderThreshold, 'max', 1, 'min', 0, 'sliderstep', [0.005 0.05], 'Value', 0.8, 'Enable', 'off');
    
    % create the listener for the sliders
    % handles.sliderListener = ...
    addlistener(handles.sliderAvgWinWidth,'ContinuousValueChange', @(hObject,eventdata) updateAvgWinWidthCallback(hObject,eventdata));
    addlistener(handles.sliderTimeAxis,'ContinuousValueChange', @(hObject,eventdata) updateImageSliderCallback(hObject,eventdata));
    addlistener(handles.sliderImageGain,'ContinuousValueChange', @(hObject,eventdata) updateImageSliderCallback(hObject,eventdata));
    addlistener(handles.sliderThreshold,'ContinuousValueChange', @(hObject,eventdata) updateImageSliderCallback(hObject,eventdata));
    
    handles.tuningGraphAxes.Color = [0 0 0];
    handles.tuningGraphAxes.Visible = 'off';    
    handles.tuningGraphAxes.DataAspectRatio = [1 1 1];
    handles.tuningGraphAxes.NextPlot = 'add';
    fill(handles.tuningGraphAxes, sin(linspace(0, 2*pi, 3600)), cos(linspace(0, 2*pi, 3600)), [0 0 0]);
    N = 16;
    plot(handles.tuningGraphAxes, flatten([0.1*ones(1, N); ones(1, N); nan(1, N)]) .* exp(1i*flatten(repmat(linspace(2*pi/N, 2*pi, N), 3, 1))), 'w:');
    handles.tuningPlotHandle = plot(handles.tuningGraphAxes, [nan nan], [nan nan], 'o', 'Color', [0.4 0.4 0.4]);
    handles.selectionTuningPlotHandle = plot(handles.tuningGraphAxes, [nan nan], [nan nan], '*', 'Color', [1.0 0.6 0.6]);
    
    % handles.rainbowOrderModeString = 'Fixed';
    % handles.rainbowAngleModeString = 'Fixed';
    % 
    % handles.toggleOrderMode.String = [handles.toggleOrderMode.String(1:strfind(handles.toggleOrderMode.String, ':')), ' ', handles.rainbowOrderModeString];
    % handles.toggleAngleMode.String = [handles.toggleAngleMode.String(1:strfind(handles.toggleAngleMode.String, ':')), ' ', handles.rainbowAngleModeString];
    
    refreshButtons(handles);
    
    % Update handles structure
    guidata(hObject, handles);
    
    % UIWAIT makes roiDetectGUI wait for user response (see UIRESUME)
    % uiwait(handles.mainFig);
    
    function handles = initialize_video_axes(handles)
    flatten = @(x) x(:);
    axes(handles.imageAxes);
    
    hObject = handles.mainFig;
    handles.imageHandle = image(double(hObject.UserData.imageStack(:, :, 1)));
    
    maxImageValue = max(flatten(hObject.UserData.imageStack));
    colormap(circshift(hot(round(double(maxImageValue))), 1, 2));
    hold on;
    
    [~, handles.thresholdingContourHandle] = contour(handles.imageHandle.CData, 'LineColor', [0.9 0.8 0], 'LineWidth', 2, 'LevelList', prctile(flatten(handles.imageHandle.CData), 50), 'Visible', 'off');
    
    % handles.overlayOfCurROI = image(ones(size(handles.imageStack, 1), size(handles.imageStack, 2)) .* permute([0.8 0.1 0.3], [1 3 2]), 'AlphaData', 0);
    % handles.overlayOfAllROIs = image(ones(size(handles.imageStack, 1), size(handles.imageStack, 2)) .* permute([1 1 1], [1 3 2]), 'AlphaData', 0);
    
    handles.overlayContourHandles = {};
    
    hObject.UserData.underSelection = image(ones(size(hObject.UserData.imageStack, 1), size(hObject.UserData.imageStack, 2)) .* permute([0.9 0.8 0.0], [1 3 2]), 'AlphaData', 0);
    
    X = hObject.UserData.originalDims(2);
    Y = hObject.UserData.originalDims(1);
    Z = hObject.UserData.originalDims(4);
    
    plot(repmat([0 X nan], 1, Z)+0.5, flatten(repmat(cumsum(Y * ones(Z, 1)), [1, 3])')' + 0.5, 'w');
    axis off;
    axis equal;
    
    
    % --- Executes on slider movement.
    function updateAvgWinWidthCallback(hObject, eventdata)
    
    handles = guidata(hObject);
    T = handles.mainFig.UserData.originalDims(3);
    t = handles.sliderTimeAxis.Value;
    b = handles.avgWidthSlider.base;
    w = round(b^handles.sliderAvgWinWidth.Value);
    if(T == w)
        handles.sliderTimeAxis.Enable = 'off';
        set(handles.sliderTimeAxis, 'max', 1.0001, 'min', 1, 'sliderstep', [0.0001 0.99999], 'Value', max(1, min(T-w+1, t)));
    else
        handles.sliderTimeAxis.Enable = 'on';
        set(handles.sliderTimeAxis, 'max', T-w+1, 'min', 1, 'sliderstep', [1 w]/(T-w), 'Value', max(1, min(T-w+1, t)));
    end
    updateImage(handles);
    guidata(hObject, handles);
    
    
    % --- Executes on slider movement.
    function updateImageSliderCallback(hObject, eventdata)
    
    handles = guidata(hObject);
    updateImage(handles);
    guidata(hObject, handles);
    
    
    function handles = updateDisplayForROIs(handles)
    
    mainFig = handles.mainFig;
    
    for i = 1:numel(handles.overlayContourHandles)
        handles.overlayContourHandles{i}.Visible = 'off';
    end
    
    for i = numel(handles.overlayContourHandles)+1:size(mainFig.UserData.ROIs, 3)
        [~, handles.overlayContourHandles{i}] = contour(mainFig.UserData.ROIs(:, :, i), 'LineColor', [0.8 0.2 0.2], 'LineWidth', 2, 'LevelList', 0.5, 'Visible', 'off', 'Parent', handles.imageAxes);
    end
    
    for vi = 1:numel(handles.listboxROIs.Value)
        v = handles.listboxROIs.Value(vi);
        
        %     if(v > numel(handles.overlayContourHandles) || isempty(handles.overlayContourHandles{v}) || ~isgraphics(handles.overlayContourHandles{v}) || ~contains(class(handles.overlayContourHandles{v}), 'Contour'))
        %         [~, handles.overlayContourHandles{v}] = contour(handles.ROIs(:, :, v), 'LineColor', [0.8 0.2 0.2], 'LineWidth', 2, 'LevelList', 0.5, 'Visible', 'off', 'Parent', handles.imageAxes);
        %     else
        handles.overlayContourHandles{v}.ZData = mainFig.UserData.ROIs(:, :, v);
        %     end
        handles.overlayContourHandles{v}.Visible = 'on';
    end
    
    handles.tuningPlotHandle.XData = real(mainFig.UserData.tuningVectors);
    handles.tuningPlotHandle.YData = imag(mainFig.UserData.tuningVectors);
    
    handles.selectionTuningPlotHandle.XData = real(mainFig.UserData.tuningVectors(handles.listboxROIs.Value));
    handles.selectionTuningPlotHandle.YData = imag(mainFig.UserData.tuningVectors(handles.listboxROIs.Value));
    
    
    
    % if(isempty(handles.ROIs))
    %     return;
    % end
    % i = handles.listboxROIs.Value;
    % 
    % handles.overlayOfCurROI.AlphaData = handles.sliderVisibilityCurrentROI.Value .* double(handles.ROIs(:, :, i));
    % handles.overlayOfAllROIs.AlphaData = handles.sliderVisibilityAllROIs.Value .* double(any(handles.ROIs, 3));
    
    
    function handles = updateImage(handles)
    flatten = @(x) x(:);
    
    mainFig = handles.mainFig;
    
    t = round(handles.sliderTimeAxis.Value);
    b = handles.avgWidthSlider.base;
    w = round(b^handles.sliderAvgWinWidth.Value);
    g = 10^handles.sliderImageGain.Value;
    
    hObject = handles.mainFig;
    
    handles.imageHandle.CData = g*mean(double(mainFig.UserData.imageStack(:, :, t:t+w-1)), 3);
    
    if(strcmp(handles.thresholdingContourHandle.Visible, 'on') == 1)
        if(handles.menuThresholdType.Value == 1 || numel(handles.listboxROIs.Value) ~= 1)
            handles.thresholdingContourHandle.ZData = handles.imageHandle.CData;
        else
            i = handles.listboxROIs.Value;
            
            activityProfileOfROI = permute(mean(mean(double(mainFig.UserData.imageStack) .* mainFig.UserData.ROIs(:, :, i), 2), 1), [3 1 2]);
            corrImage = corr(double(reshape(mainFig.UserData.imageStack, [size(mainFig.UserData.imageStack, 1)*size(mainFig.UserData.imageStack, 2), size(mainFig.UserData.imageStack, 3)])'), activityProfileOfROI);
            corrImage = reshape(corrImage, [size(mainFig.UserData.imageStack, 1), size(mainFig.UserData.imageStack, 2)]);
            if(handles.menuThresholdType.Value == 3)
                corrImage = -corrImage;
            end
            handles.thresholdingContourHandle.ZData = corrImage;
        end
        handles.thresholdingContourHandle.LevelList = prctile(flatten(handles.thresholdingContourHandle.ZData), sqrt(1 - (1-handles.sliderThreshold.Value)^2)*100);
    end
    
    % --- Outputs from this function are returned to the command line.
    function varargout = roiDetectGUI_OutputFcn(hObject, eventdata, handles) 
    % varargout  cell array for returning output args (see VARARGOUT);
    % hObject    handle to figure
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Get default command line output from handles structuren
    varargout{1} = handles.output;
    varargout{2} = hObject.UserData.ROIs;
    
    
    % --- Executes on selection change in listboxROIs.
    function listboxROIs_Callback(hObject, eventdata, handles)
    % hObject    handle to listboxROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: contents = cellstr(get(hObject,'String')) returns listboxROIs contents as cell array
    %        contents{get(hObject,'Value')} returns selected item from listboxROIs
    
    handles = guidata(hObject);
    
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    
    guidata(hObject, handles);
    
    
    % --- Executes during object creation, after setting all properties.
    function listboxROIs_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to listboxROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: listbox controls usually have a white background on Windows.
    %       See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    % --- Executes on button press in buttonDeleteROI.
    function buttonDeleteROI_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonDeleteROI (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    selection = handles.listboxROIs.Value;
    remaining = 1:numOfROIs(handles);
    remaining(selection) = [];
    handles = modifyROIs(handles, mainFig.UserData.ROIs(:, :, remaining), mainFig.UserData.tuningVectors(remaining), mainFig.UserData.identityTracker(remaining));
    refreshListBox(handles);
    handles.listboxROIs.Value = [];
    
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonDrawCircle.
    function buttonDrawCircle_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonDrawCircle (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    
    % --- Executes on button press in buttonDrawPolygon.
    function buttonDrawPolygon_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonDrawPolygon (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    if(isfield(mainFig.UserData, 'shape') && ~isempty(mainFig.UserData.shape) && isvalid(mainFig.UserData.shape))
        error('Already in the middle of a drawing');
    end
    
    disableDrawButtons(handles);
    
    mainFig.UserData.shape = impoly(handles.imageAxes, 'PositionConstraintFcn',  makeConstrainToRectFcn('impoly', handles.imageAxes.XLim, handles.imageAxes.YLim));
    
    prepareShape(handles);
    
    guidata(hObject, handles);
    
    
    function doBeforeDeletingShape(handles)
    
    handles.buttonNewROI.Enable = 'off';
    handles.buttonCancelDraw.Enable = 'off';
    
    enableDrawButtons(handles);
    
    
    % --- Executes on button press in buttonDrawFreehand.
    function buttonDrawFreehand_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonDrawFreehand (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    if(isfield(mainFig.UserData, 'shape') && ~isempty(mainFig.UserData.shape) && isvalid(mainFig.UserData.shape))
        error('Already in the middle of a drawing');
    end
    
    disableDrawButtons(handles);
    
    mainFig.UserData.shape = imfreehand(handles.imageAxes, 'PositionConstraintFcn',  makeConstrainToRectFcn('imfreehand', handles.imageAxes.XLim, handles.imageAxes.YLim));
    
    prepareShape(handles);
    
    guidata(hObject, handles);
    
    % --- Executes on button press in buttonAddShapeToROI.
    function buttonNewROI_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonAddShapeToROI (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    if(~isfield(mainFig.UserData, 'shape') || isempty(mainFig.UserData.shape) || ~isvalid(mainFig.UserData.shape))
        error('buttonAddShapeToROI_Callback: no shape exists to add to ROI.');
    end
    
    
    [X, Y] = meshgrid((1:size(mainFig.UserData.imageStack, 2)), (1:size(mainFig.UserData.imageStack, 1)));
    
    polygon = mainFig.UserData.shape.getPosition;
    isIn = inpolygon(X, Y, polygon(:, 1)', polygon(:, 2)');
    
    newIdentity = mainFig.UserData.maxIdentityNum+1;
    mainFig.UserData.maxIdentityNum = newIdentity;
    
    handles = modifyROIs(handles, cat(3, mainFig.UserData.ROIs, isIn), [mainFig.UserData.tuningVectors, 0], [mainFig.UserData.identityTracker, newIdentity]);
    refreshListBox(handles);
    handles.listboxROIs.Value = numOfROIs(handles);
    
    delete(mainFig.UserData.shape);
    refreshButtons(handles);
    
    handles = updateDisplayForROIs(handles);
    
    guidata(hObject, handles);
    
    
    
    % --- Executes on button press in buttonThreshold.
    function buttonThreshold_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonThreshold (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    disableDrawButtons(handles);
    
    mainFig = handles.mainFig;
    
    handles.thresholdingContourHandle.Visible = 'on';
    mainFig.UserData.underSelection.Visible = 'on';
    
    handles.sliderThreshold.Enable = 'on';
    handles.menuThresholdType.Enable = 'on';
    updateImage(handles);
    
    handles.imageAxes.Parent.WindowButtonMotionFcn = @mouseMoveDuringThresholdSelection;
    handles.imageAxes.Parent.WindowButtonDownFcn = @mouseClickDuringThresholdSelection;
    
    guidata(hObject, handles);
    
    function mouseClickDuringThresholdSelection(hObject, eventdata)
        
    % TODO check to see if threshold selection is still ongoing
    
    handles = guidata(hObject);
    mainFig = handles.mainFig;
    
    cp = get (gca, 'CurrentPoint');
    x = cp(1, 1);
    y = cp(1, 2);
    
    polygonPoints = findPolygonFromContourMatrixThatContains(x, y, handles.thresholdingContourHandle.ContourMatrix);
    
    mainFig.UserData.shape = impoly(handles.imageAxes, polygonPoints', 'PositionConstraintFcn',  makeConstrainToRectFcn('impoly', handles.imageAxes.XLim, handles.imageAxes.YLim));
    
    handles.thresholdingContourHandle.Visible = 'off';
    mainFig.UserData.underSelection.Visible = 'off';
    
    handles.sliderThreshold.Enable = 'off';
    handles.menuThresholdType.Enable = 'off';
    
    prepareShape(handles);
    
    handles.imageAxes.Parent.WindowButtonMotionFcn = [];
    handles.imageAxes.Parent.WindowButtonDownFcn = [];
    
    guidata(hObject, handles);
    
    function mouseMoveDuringThresholdSelection(hObject, eventdata)
    
    
    % TODO check to see if threshold selection is still ongoing
    
    handles = guidata(hObject);
    mainFig = handles.mainFig;
    
    cp = get (gca, 'CurrentPoint');
    x = cp(1, 1);
    y = cp(1, 2);
    
    polygonPoints = findPolygonFromContourMatrixThatContains(x, y, handles.thresholdingContourHandle.ContourMatrix);
    
    [X, Y] = meshgrid((1:size(mainFig.UserData.imageStack, 2)), (1:size(mainFig.UserData.imageStack, 1)));
    
    if(~isempty(polygonPoints))
        mainFig.UserData.underSelection.AlphaData = 0.8 * double(inpolygon(X, Y, polygonPoints(1, :), polygonPoints(2, :)));
    else
        mainFig.UserData.underSelection.AlphaData = 0;
    end
    
    guidata(hObject, handles);
    
    
    
    function polygonPoints = findPolygonFromContourMatrixThatContains(x, y, contourMatrix)
    
    i = 1;
    while(i < size(contourMatrix, 2))
    n = contourMatrix(2, i);
    polygonPoints = contourMatrix(1:2, i+1:i+n);
    if(inpolygon(x, y, polygonPoints(1, :), polygonPoints(2, :)))
        return;
    end
    i = i+n+1;
    end
    
    polygonPoints = [];
    
    
    function prepareShape(handles)
    
    mainFig = handles.mainFig;
    
    mainFig.UserData.shape.setColor([0.9725 0.3098 0.3098]);
    addlistener(mainFig.UserData.shape,'ObjectBeingDestroyed', @(~, ~) doBeforeDeletingShape(handles));
    refreshButtons(handles);
    
    function disableDrawButtons(handles)
    
    handles.buttonDrawCircle.Enable = 'off';
    handles.buttonDrawFreehand.Enable = 'off';
    handles.buttonDrawPolygon.Enable = 'off';
    handles.buttonThreshold.Enable = 'off';
    
    function enableDrawButtons(handles)
    
    handles.buttonDrawCircle.Enable = 'on';
    handles.buttonDrawFreehand.Enable = 'on';
    handles.buttonDrawPolygon.Enable = 'on';
    handles.buttonThreshold.Enable = 'on';
    
    function result = numOfROIs(handles)
        result = size(handles.mainFig.UserData.ROIs, 3);
    
    function refreshListBox(handles)
    
    if(numel(handles.listboxROIs.String) > numOfROIs(handles))
        handles.listboxROIs.String = handles.listboxROIs.String(1:numOfROIs(handles));
    end
    
    for i = 1:numOfROIs(handles)
        handles.listboxROIs.String{i} = sprintf('ROI %03d', i);
    end
    
    handles.listboxROIs.Value(handles.listboxROIs.Value > numel(handles.listboxROIs.String)) = [];
    
    
    function refreshButtons(handles)
    
    % if(numel(handles.listboxROIs.Value) == 1)
    %     handles.buttonDeleteROI.Enable = 'on';
    %     handles.buttonDuplicateROI.Enable = 'on';
    %     handles.buttonIntersectROIs.Enable = 'off';
    %     handles.buttonMergeROIs.Enable = 'off';
    %     handles.buttonFragmentROIs.Enable = 'off';
    % elseif(numel(handles.listboxROIs.Value) > 1)
    %     handles.buttonDeleteROI.Enable = 'off';
    %     handles.buttonDuplicateROI.Enable = 'off';
    %     handles.buttonIntersectROIs.Enable = 'on';
    %     handles.buttonMergeROIs.Enable = 'on';
    %     handles.buttonFragmentROIs.Enable = 'on';
    % else
    %     handles.buttonDeleteROI.Enable = 'off';
    %     handles.buttonDuplicateROI.Enable = 'off';
    %     handles.buttonIntersectROIs.Enable = 'off';
    %     handles.buttonMergeROIs.Enable = 'off';
    %     handles.buttonFragmentROIs.Enable = 'off';
    % end
    
    handles.buttonDeleteROI.Enable = 'off';
    handles.buttonDuplicateROI.Enable = 'off';
    handles.buttonIntersectROIs.Enable = 'off';
    handles.buttonMergeROIs.Enable = 'off';
    handles.buttonFragmentROIs.Enable = 'off';
    handles.buttonSwapDown.Enable = 'off';
    handles.buttonSwapUp.Enable = 'off';
    handles.buttonUndo.Enable = 'off';
    handles.buttonRedo.Enable = 'off';
    
    handles.buttonUp.Enable = 'on';
    handles.buttonDown.Enable = 'on';
    handles.buttonLeft.Enable = 'on';
    handles.buttonRight.Enable = 'on';
    
    mainFig = handles.mainFig;
    
    if(numel(handles.listboxROIs.Value) == 1)
        handles.buttonDeleteROI.Enable = 'on';
        handles.buttonDuplicateROI.Enable = 'on';
        
        if(handles.listboxROIs.Value < numOfROIs(handles))
            handles.buttonSwapDown.Enable = 'on';
        end
        
        if(handles.listboxROIs.Value > 1)
            handles.buttonSwapUp.Enable = 'on';
        end
        
    elseif(numel(handles.listboxROIs.Value) > 1)
        handles.buttonDeleteROI.Enable = 'on';
        handles.buttonIntersectROIs.Enable = 'on';
        handles.buttonMergeROIs.Enable = 'on';
        handles.buttonFragmentROIs.Enable = 'on';
    else
        handles.buttonUp.Enable = 'off';
        handles.buttonDown.Enable = 'off';
        handles.buttonLeft.Enable = 'off';
        handles.buttonRight.Enable = 'off';
    end
    
    if(isfield(mainFig.UserData, 'shape') && ~isempty(mainFig.UserData.shape) && isvalid(mainFig.UserData.shape))
        handles.buttonNewROI.Enable = 'on';
        handles.buttonCancelDraw.Enable = 'on';
    else
        handles.buttonNewROI.Enable = 'off';
        handles.buttonCancelDraw.Enable = 'off';
    end
    
    if(mainFig.UserData.historyIndex > 1)
        handles.buttonUndo.Enable = 'on';
    % else
    %     handles.buttonUndo.Enable = 'off';
    end
    
    if(mainFig.UserData.historyIndex < numel(mainFig.UserData.history))
        handles.buttonRedo.Enable = 'on';
    % else
    %     handles.buttonRedo.Enable = 'off';
    end
    
    function handles = modifyROIs(handles, newROIs, newTuningVectors, newIDs)
    
    mainFig = handles.mainFig;
    
    if(numel(mainFig.UserData.history) > mainFig.UserData.historyIndex)
        mainFig.UserData.history(mainFig.UserData.historyIndex+1:end) = [];
    end
    assert(numel(mainFig.UserData.history) == mainFig.UserData.historyIndex);
    
    mainFig.UserData.ROIs = double(newROIs);
    mainFig.UserData.tuningVectors = newTuningVectors;
    mainFig.UserData.identityTracker = newIDs;
    
    mainFig.UserData.historyIndex = mainFig.UserData.historyIndex+1;
    mainFig.UserData.history(mainFig.UserData.historyIndex).ROIs = mainFig.UserData.ROIs;
    mainFig.UserData.history(mainFig.UserData.historyIndex).tuningVectors = mainFig.UserData.tuningVectors;
    mainFig.UserData.history(mainFig.UserData.historyIndex).identityTracker = mainFig.UserData.identityTracker;
    
    function handles = undoROIs(handles)
    mainFig = handles.mainFig;
    if(mainFig.UserData.historyIndex == 1)
        error('Cannot undo anymore.');
        return;
    end
    mainFig.UserData.historyIndex = mainFig.UserData.historyIndex-1;
    mainFig.UserData.ROIs = mainFig.UserData.history(mainFig.UserData.historyIndex).ROIs;
    mainFig.UserData.tuningVectors = mainFig.UserData.history(mainFig.UserData.historyIndex).tuningVectors;
    mainFig.UserData.identityTracker = mainFig.UserData.history(mainFig.UserData.historyIndex).identityTracker;
    
    function handles = redoROIs(handles)
    mainFig = handles.mainFig;
    if(mainFig.UserData.historyIndex == numel(mainFig.UserData.history))
        error('Cannot redo.');
        return;
    end
    mainFig.UserData.historyIndex = mainFig.UserData.historyIndex+1;
    mainFig.UserData.ROIs = mainFig.UserData.history(mainFig.UserData.historyIndex).ROIs;
    mainFig.UserData.tuningVectors = mainFig.UserData.history(mainFig.UserData.historyIndex).tuningVectors;
    mainFig.UserData.identityTracker = mainFig.UserData.history(mainFig.UserData.historyIndex).identityTracker;
    
    % --- Executes on button press in buttonCancelDraw.
    function buttonCancelDraw_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonCancelDraw (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    mainFig = handles.mainFig;
    delete(mainFig.UserData.shape);
    
    guidata(hObject, handles);
    
    
    % --- Executes on slider movement.
    function sliderAvgWinWidth_Callback(hObject, eventdata, handles)
    % hObject    handle to sliderAvgWinWidth (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'Value') returns position of slider
    %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
    
    
    % --- Executes during object creation, after setting all properties.
    function sliderAvgWinWidth_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to sliderAvgWinWidth (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end
    
    
    % --- Executes on slider movement.
    function sliderVisibilityCurrentROI_Callback(hObject, eventdata, handles)
    % hObject    handle to sliderVisibilityCurrentROI (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'Value') returns position of slider
    %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
    
    
    % --- Executes during object creation, after setting all properties.
    function sliderVisibilityCurrentROI_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to sliderVisibilityCurrentROI (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end
    
    
    % --- Executes on slider movement.
    function sliderVisibilityAllROIs_Callback(hObject, eventdata, handles)
    % hObject    handle to sliderVisibilityAllROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'Value') returns position of slider
    %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
    
    
    % --- Executes during object creation, after setting all properties.
    function sliderVisibilityAllROIs_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to sliderVisibilityAllROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end
    
    
    % --- Executes on slider movement.
    function sliderImageGain_Callback(hObject, eventdata, handles)
    % hObject    handle to sliderImageGain (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'Value') returns position of slider
    %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
    
    
    % --- Executes during object creation, after setting all properties.
    function sliderImageGain_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to sliderImageGain (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end
    
    % --- Executes on slider movement.
    function sliderThreshold_Callback(hObject, eventdata, handles)
    % hObject    handle to sliderThreshold (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'Value') returns position of slider
    %        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
    
    
    % --- Executes during object creation, after setting all properties.
    function sliderThreshold_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to sliderThreshold (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end
    
    
    % --- Executes on button press in buttonMergeROIs.
    function buttonMergeROIs_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonMergeROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    selection = handles.listboxROIs.Value;
    theOthers = 1:numOfROIs(handles);
    if(handles.checkboxAutoDelete.Value == 1)
        theOthers(selection) = [];
    end
    
    % totalPixelsPerSelectedROI = permute(sum(sum(handles.ROIs(:, :, selection) , 1), 2), [1 3 2]);
    % mergedTuningVector = sum(handles.tuningVectors(selection) .* totalPixelsPerSelectedROI) ./ sum(totalPixelsPerSelectedROI);
    % % we take the weighted mean vector of the merging ROIs weighted by their
    % % relative number of pixels. This is an approximation to the mathematically
    % % correct thing to do which is to used standard deviatoin of each ROIs
    % % activity profile to calculate its corresponding alpha and beta and then
    % % calculate signal quality from that
    
    mainFig = handles.mainFig;
    
    newIdentity = mainFig.UserData.maxIdentityNum+1;
    mainFig.UserData.maxIdentityNum = newIdentity;
    
    handles = modifyROIs(handles, cat(3, mainFig.UserData.ROIs(:, :, theOthers), max(mainFig.UserData.ROIs(:, :, selection), [], 3)), [mainFig.UserData.tuningVectors(theOthers) 0], [mainFig.UserData.identityTracker(theOthers) newIdentity]);
    handles.listboxROIs.Value = numOfROIs(handles);
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonIntersectROIs.
    function buttonIntersectROIs_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonIntersectROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    newIdentity = mainFig.UserData.maxIdentityNum+1;
    mainFig.UserData.maxIdentityNum = newIdentity;
    
    selection = handles.listboxROIs.Value;
    theOthers = 1:numOfROIs(handles);
    if(handles.checkboxAutoDelete.Value == 1)
        theOthers(selection) = [];
    end
    
    handles = modifyROIs(handles, cat(3, mainFig.UserData.ROIs(:, :, theOthers), min(mainFig.UserData.ROIs(:, :, selection), [], 3)), [mainFig.UserData.tuningVectors(theOthers) 0], [mainFig.UserData.identityTracker(theOthers) newIdentity]);
    handles.listboxROIs.Value = numOfROIs(handles);
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonFragmentROIs.
    function buttonFragmentROIs_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonFragmentROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    flatten = @(x) x(:);
    mainFig = handles.mainFig;
    
    selection = handles.listboxROIs.Value;
    addTheseROIs = [];
    for i = 1:(2^numel(selection)-1)
        coefficientVector = permute((dec2bin(i, numel(selection))-'0'), [1 3 2]);
        m = min((1-coefficientVector) + (coefficientVector*2 - 1) .* mainFig.UserData.ROIs(:, :, selection), [], 3);
        if(sum(flatten(m)) > 1 && sum(flatten(m > 0)) > 1)
            addTheseROIs(:, :, end+1) = m;
        end
    end
    
    newIdentities = mainFig.UserData.maxIdentityNum+(1:(size(addTheseROIs, 3)+1));
    mainFig.UserData.maxIdentityNum = newIdentities(end);
    
    theOthers = 1:numOfROIs(handles);
    if(handles.checkboxAutoDelete.Value == 1)
        theOthers(selection) = [];
    end
    handles = modifyROIs(handles, cat(3, mainFig.UserData.ROIs(:, :, theOthers), addTheseROIs), [mainFig.UserData.tuningVectors(theOthers) zeros(1, size(addTheseROIs, 3))], [mainFig.UserData.identityTracker(theOthers) newIdentities]);
    handles.listboxROIs.Value = numel(theOthers)+1:numOfROIs(handles);
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    guidata(hObject, handles);
    
    
    
    % --- Executes on button press in buttonDuplicateROI.
    function buttonDuplicateROI_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonDuplicateROI (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    newIdentity = mainFig.UserData.maxIdentityNum+1;
    mainFig.UserData.maxIdentityNum = newIdentity;
    
    i = handles.listboxROIs.Value;
    handles = modifyROIs(handles, mainFig.UserData.ROIs(:, :, [1:i-1, i, i, i+1:end]), mainFig.UserData.tuningVectors([1:i-1, i, i, i+1:end]), [mainFig.UserData.identityTracker(1:i), newIdentity, mainFig.UserData.identityTracker(i+1:end)]);
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonSaveROIs.
    function buttonSaveROIs_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonSaveROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    ROIs = mainFig.UserData.ROIs;
    
    if(~exist('.backupROIs', 'dir'))
        mkdir .backupROIs;
    end
    
    tuningVectors = mainFig.UserData.tuningVectors;
    if(~isfield(mainFig.UserData, 'fileName') || isempty(mainFig.UserData.fileName))
        fprintf('Saving to ROIs.mat...');
        save ROIs.mat ROIs tuningVectors;
        fprintf('[DONE]\n');
        fprintf(['\tAlso saving to .backupROIs/ROI__', datestr(now, 'yyyy_mm_dd_HH_MM'), '.mat...']);
        save(['.backupROIs/ROIs_', datestr(now, 'yyyy_mm_dd_HH_MM'), '.mat'], 'ROIs', 'tuningVectors');
        fprintf('[DONE]\n');
    else
        saveToThisFile = strrep(mainFig.UserData.fileName, '.mat', '.ROIs.mat');
        fprintf('Saving to %s...', saveToThisFile);
        if(exist(saveToThisFile, 'file'))
            save(saveToThisFile, 'ROIs', 'tuningVectors', '-append');
        else
            save(saveToThisFile, 'ROIs', 'tuningVectors');
        end
        fprintf('[DONE]\n');
        fprintf(['\tAlso saving to .backupROIs/ROI_', strrep(mainFig.UserData.fileName, '.mat', ''), '_', datestr(now, 'yyyy_mm_dd_HH_MM'), '.mat...']);
        save(['.backupROIs/ROI_', strrep(mainFig.UserData.fileName, '.mat', ''), '_', datestr(now, 'yyyy_mm_dd_HH_MM'), '.mat'], 'ROIs', 'tuningVectors');
        fprintf('[DONE]\n');
    end
    
    
    % --- Executes on button press in loadROIs.
    function loadROIs_Callback(hObject, eventdata, handles)
    % hObject    handle to loadROIs (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    uiopen('.backupROIs/*.mat')
    
    if(~exist('ROIs', 'var'))
        error('File did not contain an ROIs variable');
    end
    
    ROIs = ROIs ./ max(1, max(max(ROIs, [], 1), [], 2)); % some of the ROIs where saved with large numbers due to a bug
    
    if(~exist('tuningVectors', 'var') || isempty(tuningVectors))
        tuningVectors = zeros(1, size(ROIs, 3));
    end
    
    mainFig = handles.mainFig;
    
    if(size(ROIs, 1) < size(mainFig.UserData.imageStack, 1) || size(ROIs, 2) < size(mainFig.UserData.imageStack, 2))
        warning('ROIs were created for a smaller image stack.');
        ROIs(end+1:size(mainFig.UserData.imageStack, 1), end+1:size(mainFig.UserData.imageStack, 2), :) = 0;
    end
    
    if(size(ROIs, 1) > size(mainFig.UserData.imageStack, 1) || size(ROIs, 2) > size(mainFig.UserData.imageStack, 2))
        warning('Cropping ROIs to fit the image stack.');
        ROIs(size(mainFig.UserData.imageStack, 1)+1:end, :, :) = [];
        ROIs(:, size(mainFig.UserData.imageStack, 2)+1:end, :) = [];
    end
    
    newIdentities = mainFig.UserData.maxIdentityNum + (1:(size(ROIs, 3)+1));
    mainFig.UserData.maxIdentityNum = newIdentities(end);
    
    handles = modifyROIs(handles, ROIs, tuningVectors, newIdentities);
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    
    guidata(hObject, handles);
    
    
    % --- Executes during object creation, after setting all properties.
    function slider1_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to slider1 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: slider controls usually have a light gray background.
    if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor',[.9 .9 .9]);
    end
    
    % --- Executes during object creation, after setting all properties.
    function slider1_Callback(hObject, eventdata, handles)
    % hObject    handle to slider1 (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    
    % --- Executes on button press in buttonSwapUp.
    function buttonSwapUp_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonSwapUp (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    i = handles.listboxROIs.Value;
    handles = modifyROIs(handles, mainFig.UserData.ROIs(:, :, [1:i-2, i, i-1, i+1:end]), mainFig.UserData.tuningVectors([1:i-2, i, i-1, i+1:end]), mainFig.UserData.identityTracker([1:i-2, i, i-1, i+1:end]));
    handles.listboxROIs.Value = i-1;
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonSwapDown.
    function buttonSwapDown_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonSwapDown (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    i = handles.listboxROIs.Value;
    handles = modifyROIs(handles, mainFig.UserData.ROIs(:, :, [1:i-1, i+1, i, i+2:end]), mainFig.UserData.tuningVectors([1:i-1, i+1, i, i+2:end]), mainFig.UserData.identityTracker([1:i-1, i+1, i, i+2:end]));
    handles.listboxROIs.Value = i+1;
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    guidata(hObject, handles);
    
    
    
    % --- Executes on button press in buttonUndo.
    function buttonUndo_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonUndo (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    handles = undoROIs(handles);
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonRedo.
    function buttonRedo_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonRedo (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    handles = redoROIs(handles);
    refreshListBox(handles);
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    guidata(hObject, handles);
    
    % --- Executes on button press in checkboxAutoDelete.
    function checkboxAutoDelete_Callback(hObject, eventdata, handles)
    % hObject    handle to checkboxAutoDelete (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hint: get(hObject,'Value') returns toggle state of checkboxAutoDelete
    
    % --- Executes on selection change in menuThresholdType.
    function menuThresholdType_Callback(hObject, eventdata, handles)
    % hObject    handle to menuThresholdType (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: contents = cellstr(get(hObject,'String')) returns menuThresholdType contents as cell array
    %        contents{get(hObject,'Value')} returns selected item from menuThresholdType
    
    handles = updateImage(handles);
    
    guidata(hObject, handles);
    
    % --- Executes during object creation, after setting all properties.
    function menuThresholdType_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to menuThresholdType (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: popupmenu controls usually have a white background on Windows.
    %       See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    
    function handles = shiftSelectedROIs(handles, shiftVector)
    
    selection = handles.listboxROIs.Value;
    movedROIs = handles.mainFig.UserData.ROIs;
    movedROIs(:, :, selection) = circshift(movedROIs(:, :, selection), shiftVector);
    handles = modifyROIs(handles, movedROIs, handles.mainFig.UserData.tuningVectors, handles.mainFig.UserData.identityTracker);
    handles = updateDisplayForROIs(handles);
    
    
    
    % --- Executes on button press in buttonUp.
    function buttonUp_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonUp (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    handles = shiftSelectedROIs(handles, [-1 0]*1);
    guidata(hObject, handles);
    
    
    
    % --- Executes on button press in buttonDown.
    function buttonDown_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonDown (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    handles = shiftSelectedROIs(handles, [1 0]*1);
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonLeft.
    function buttonLeft_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonLeft (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    handles = shiftSelectedROIs(handles, [0 -1]*1);
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonRight.
    function buttonRight_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonRight (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    handles = shiftSelectedROIs(handles, [0 1]*1);
    guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonRainbow.
    function buttonRainbow_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonRainbow (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)P
        
    flatten = @(x) x(:);
    
    wb = waitbar(0, 'Calculating activity profiles...');
    
    % activityProfileOfROIs = zeros(size(handles.imageStack, 3), size(handles.ROIs, 3));
    % for i_r = 1:size(handles.ROIs, 3)
    %     activityProfileOfROIs(:, i_r) = permute(mean(mean(double(handles.imageStack) .* handles.ROIs(:, :, i_r), 2), 1), [3 1 2]);
    %     wb = waitbar(i_r/size(handles.ROIs, 3), wb, 'Calculating activity profiles...');
    % end
    
    mainFig = handles.mainFig;
    
    activityProfileOfROIs = calculateActivityProfiles(handles, wb);
    
    waitbar(0, wb, 'Correlating pixels with activity profiles...');
    C = zeros(size(mainFig.UserData.imageStack, 1), size(mainFig.UserData.imageStack, 2), size(mainFig.UserData.ROIs, 3));
    
    for i_r = 1:size(mainFig.UserData.ROIs, 3)
        corrImage = corr(double(reshape(mainFig.UserData.imageStack, [size(mainFig.UserData.imageStack, 1)*size(mainFig.UserData.imageStack, 2), size(mainFig.UserData.imageStack, 3)])'), activityProfileOfROIs(:, i_r));
        C(:, :, i_r) = reshape(corrImage, [size(mainFig.UserData.imageStack, 1), size(mainFig.UserData.imageStack, 2)]);
        wb = waitbar(i_r/size(mainFig.UserData.ROIs, 3), wb, 'Correlating pixels with activity profiles...');
    end
    
    [~, reorderROIs] = sort(angle(mainFig.UserData.tuningVectors));
    
    invalidROIs = find(flatten(sum(sum(mainFig.UserData.ROIs, 1), 2) == 0) | flatten(mainFig.UserData.tuningVectors == 0));
    reorderROIs(reorderROIs == invalidROIs) = [];
    
    C = C(:, :, reorderROIs);
    tuningAngles = permute(angle(mainFig.UserData.tuningVectors), [1 3 2]);
    tuningAngles = tuningAngles(reorderROIs);
    
    tuningMagnitudes = permute(abs(mainFig.UserData.tuningVectors), [1 3 2]);
    tuningMagnitudes = tuningMagnitudes(reorderROIs);
    
    waitbar(0, wb, 'calculating FFT...');
    fts = fft(C, [], 3);
    
    mainFig.UserData.rainbowImageFFT = fts(:, :, 2) / size(C, 3) * 2;
    
    hues = mod(angle(mainFig.UserData.rainbowImageFFT)-tuningAngles(1), 2*pi) / (2*pi);
    vals = abs(mainFig.UserData.rainbowImageFFT);
    % vals = vals ./ (1 - vals.^2) ./ std(mainFig.UserData.imageStack, 0, 3);
    % vals = vals ./ max(vals(:));
    % sats = 1 - sqrt(mean((vals .* cos(alphas + hues * 2*pi) - C).^2, 3));
    sats = 1 - mean((tuningMagnitudes .* cos(tuningAngles + hues * 2*pi) - C).^2, 3);
    
    handles.rainbowFigure = figure('units','normalized','outerposition',[0.1 0.1 0.5 0.9], 'Color', [0 0 0]);
    rainbowImageM = hsv2rgb(cat(3, hues, sats, vals ./ max(flatten(vals))));
    
    N = 16;
    
    subplot(5, 2, 1:2:9);
    rainbowImageHandle = image(rainbowImageM);
    axis off;
    axis equal;
    hold on;
    
    imageDimX = mainFig.UserData.originalDims(2);
    imageDimY = mainFig.UserData.originalDims(1);
    imageDimZ = mainFig.UserData.originalDims(4);
    
    imageDims = [imageDimX, imageDimY, imageDimZ];
    
    plot(repmat([0 imageDimX nan], 1, imageDimZ-1) + 0.5, flatten(repmat(cumsum(imageDimY * ones(imageDimZ-1, 1)), [1, 3])')' + 0.5, 'w');
    plot([0 0 1 1 0] * imageDimX + 0.5, [0 1 1 0 0] * imageDimY * imageDimZ + 0.5, 'w');
    
    handles.rainbowContourCoreHandles = {};
    for i = 1:N
        [~, handles.rainbowContourCoreHandles{i}] = contour(vals, 'Visible', 'off');
    end
    
    handles.rainbowContourThinHandles = {};
    for i = 1:N
        [~, handles.rainbowContourThinHandles{i}] = contour(vals, 'Visible', 'off', 'UserData', true);
    end
    
    handles.polarAxes = subplot(5, 2, 2:2:6);
    hold on;
    plot(handles.polarAxes, cos(linspace(0, 2*pi, 1000))*0.5, sin(linspace(0, 2*pi, 1000))*0.5, 'w:');
    plot(handles.polarAxes, cos(linspace(0, 2*pi, 1000)), sin(linspace(0, 2*pi, 1000)), 'w-');
    plot(handles.polarAxes, flatten([0.1*ones(1, N); ones(1, N); nan(1, N)]) .* exp(1i*flatten(repmat(linspace(2*pi/N, 2*pi, N), 3, 1))), 'w:');
    axis off;
    axis equal;
    title('Adjust ROI seed Kernels below', 'Color', 'w');
    
    handles.kernelHandles = {};
    for i = 1:N
        handles.kernelHandles{i} = plot(handles.polarAxes, [nan nan], [nan nan]);
        handles.kernelHandles{i}.UserData = [0.6, 1/40];
    end
    
    waitbar(0.1, wb, 'initializing kernels...');
    
    handles.peakPointHandles = {};
    handles.aidPointHandles = {};
    handles.seedPointHandles = {};
    for i = 1:N
        handles.peakPointHandles{i} = impoint(handles.polarAxes, 1, 0);
        handles.peakPointHandles{i}.Deletable = false;
        
        handles.aidPointHandles{i} = impoint(handles.polarAxes, 1, 0);
        handles.aidPointHandles{i}.Deletable = false;
        
        handles.seedPointHandles{i} = impoint(rainbowImageHandle.Parent, 1, 1);
        handles.seedPointHandles{i}.Deletable = false;
        
        setPositionConstraintFcn(handles.peakPointHandles{i}, @(p) p ./ max(0.99, sqrt(sum(p.^2))));
        setPositionConstraintFcn(handles.aidPointHandles{i}, @(p) p ./ max(0.99, sqrt(sum(p.^2))));
        setPositionConstraintFcn(handles.seedPointHandles{i}, makeConstrainToRectFcn('impoint', rainbowImageHandle.Parent.XLim, rainbowImageHandle.Parent.YLim));
        
    %     addNewPositionCallback(handles.peakPointHandles{i}, @(~) updateKernalAndContours(handles.kernelHandles{i}, contourHandles{i}, handles.peakPointHandles{i}, handles.aidPointHandles{i}, hues, vals));
        
        addNewPositionCallback(handles.seedPointHandles{i}, @(~) moveAlongPeak(handles.seedPointHandles{i}, handles.peakPointHandles{i}, hues));
        addNewPositionCallback(handles.peakPointHandles{i}, @(~) moveAlongAid(handles.peakPointHandles{i}, handles.aidPointHandles{i}, handles.kernelHandles{i}));
        addNewPositionCallback(handles.aidPointHandles{i}, @(~) updateKernalAndContours(handles.kernelHandles{i}, handles.rainbowContourCoreHandles{i}, handles.rainbowContourThinHandles{i}, handles.peakPointHandles{i}, handles.aidPointHandles{i}, handles.seedPointHandles{i}, hues, vals, imageDims));
        
        handles.peakPointHandles{i}.setPosition(0.9 * cos((i-1)*2*pi/N), 0.9 * sin((i-1)*2*pi/N));
        pause(0.1);
    end
    
    waitbar(1, wb, 'adding buttons...');
    
    handles.rainbowButtonHideAll = uicontrol(...
        'Parent', handles.rainbowFigure, ...
        'Style', 'pushbutton', ...
        'String', 'Hide All', ...
        'Units', 'normalized', ...
        'Position', [0.70 0.40 0.08 0.03], ...
        'Visible', 'on', ...
        'Callback', @(h, e) arrayfun(@(x) set(x{1}, 'Visible', 'off'), [handles.rainbowContourCoreHandles, handles.rainbowContourThinHandles]));
    
    handles.rainbowButtonRefreshAll = uicontrol(...
        'Parent', handles.rainbowFigure, ...
        'Style', 'pushbutton', ...
        'String', 'Refresh / Show All', ...
        'Units', 'normalized', ...
        'Position', [0.55 0.40 0.14 0.03], ...
        'Visible', 'on', ...
        'Callback', @(h, e) arrayfun(@(x) x{1}.setPosition(x{1}.getPosition), handles.peakPointHandles));
    
    selectFromCell = @(a, i) a{i};
    handles.rainbowButtonToggleThin = uicontrol(...
        'Parent', handles.rainbowFigure, ...
        'Style', 'pushbutton', ...
        'String', 'Toggle Thin Contours', ...
        'Units', 'normalized', ...
        'Position', [0.60 0.30 0.18 0.03], ...
        'Visible', 'on', ...
        'Callback', @(h, e) arrayfun(@(x) set(x{1}, 'UserData', ~x{1}.UserData(1), 'Visible', selectFromCell({'off', x{1}.Visible}, 2-x{1}.UserData(1))), handles.rainbowContourThinHandles));
    
    
    % handles.rainbowButtonMemPlus = uicontrol(...
    %     'Parent', handles.rainbowFigure, ...
    %     'Style', 'pushbutton', ...
    %     'String', 'M+', ...
    %     'Units', 'normalized', ...
    %     'Position', [0.80 0.40 0.04 0.03], ...
    %     'Visible', 'on', ...
    %     'Callback', @(h, e) saveKernels(handles));
    % 
    % 
    % handles.rainbowButtonMemRecall = uicontrol(...
    %     'Parent', handles.rainbowFigure, ...
    %     'Style', 'pushbutton', ...
    %     'String', 'MR', ...
    %     'Units', 'normalized', ...
    %     'Position', [0.85 0.40 0.04 0.03], ...
    %     'Visible', 'on', ...
    %     'Callback', @(h, e) loadKernels(handles));
    
    handles.rainbowButtonAccept = uicontrol(...
        'Parent', handles.rainbowFigure, ...
        'Style', 'pushbutton', ...
        'String', 'Accept', ...
        'Units', 'normalized', ...
        'Position', [0.75 0.20 0.08 0.03], ...
        'Visible', 'on', ...
        'Callback', @(h, e) rainbowButtonAccept_Callback(h, e, hObject, handles));
    
    handles.rainbowButtonCancel = uicontrol(...
        'Parent', handles.rainbowFigure, ...
        'Style', 'pushbutton', ...
        'String', 'Cancel', ...
        'Units', 'normalized', ...
        'Position', [0.85 0.20 0.08 0.03], ...
        'Visible', 'on', ...
        'Callback', @(h, e) rainbowButtonCancel_Callback(h, e, handles));
    
    
    close(wb);
    
    
    guidata(handles.rainbowButtonCancel, handles);
    guidata(handles.rainbowButtonAccept, handles);
    
    guidata(hObject, handles);
    
    function moveAlongPeak(seedPointHandle, peakPointHandle, hues)
    
    getMag = @(p) sqrt(sum(p.^2));
    p = getPosition(seedPointHandle);
    pixelHue = hues(round(p(2)), round(p(1)));
    
    peakPointMag = getMag(peakPointHandle.getPosition);
    
    peakPointHandle.setPosition(peakPointMag * cos(pixelHue*2*pi), peakPointMag * sin(pixelHue*2*pi));
    
    
    function moveAlongAid(peakPointHandle, aidPointHandle, kernelHandle)
    
    getHue = @(p) mod(atan2(p(2) , p(1)), 2*pi)/(2*pi);
    getMag = @(p) sqrt(sum(p.^2));
    
    aidPointMag = getMag(peakPointHandle.getPosition) * kernelHandle.UserData(1);
    aidPointHue = getHue(peakPointHandle.getPosition) + kernelHandle.UserData(2);
    
    aidPointHandle.setPosition(aidPointMag * cos(aidPointHue*2*pi), aidPointMag * sin(aidPointHue*2*pi));
    
    
    % function saveKernels(handles)
    % 
    % kernelData = {};
    % 
    % for i = 1:numel(handles.kernelHandles)
    %     kernelData{i} = handles.kernelHandles{i}.UserData;
    % end
    % 
    % kernelPeakPositions = {};
    % 
    % for i = 1:numel(handles.kernelHandles)
    %     kernelPeakPositions{i} = handles.peakPointHandles{i}.getPosition();
    % end
    % 
    % save('rainbowKernels.mat', 'kernelData', 'kernelPeakPositions');
    % save(['rainbowKernels_backup_', datestr(now, 'yyyy_mm_dd_HH_MM_SS'), '.mat'], 'kernelData', 'kernelPeakPositions');
    % 
    % 
    % function loadKernels(handles)
    % 
    % load('rainbowKernels.mat', 'kernelData', 'kernelPeakPositions');
    % 
    % for i = 1:numel(handles.kernelHandles)
    %     handles.kernelHandles{i}.UserData = kernelData{i};
    % end
    % 
    % for i = 1:numel(handles.kernelHandles)
    %     handles.peakPointHandles{i}.setPosition(kernelPeakPositions{i});
    % end
    
    
    
    function updateKernalAndContours(kernelHandle, contourCoreHandle, contourThinHandle, peakPointHandle, aidPointHandle, seedPointHandle, hues, vals, imageOriginalDims)
    
    getHue = @(p) mod(atan2(p(2) , p(1)), 2*pi)/(2*pi);
    getMag = @(p) min(1-1e-8, sqrt(sum(p.^2)));
    hueDiff = @(deltaH) (1-cos((2*pi)*(deltaH)))/2;
    sigmoid = @(x) 1 ./ (1 + exp(-x));
    flatten = @(x) x(:);
    
    color = hsv2rgb(getHue(peakPointHandle.getPosition), 0.5, 1);
    
    theta = linspace(0, 2*pi, 5*360+1);
    
    b = -log(1 ./ getMag(peakPointHandle.getPosition) - 1);
    a = (- b - log(1 ./ getMag(aidPointHandle.getPosition) - 1)) ./ hueDiff(getHue(aidPointHandle.getPosition) - getHue(peakPointHandle.getPosition));
    
    kernelHandle.UserData = [getMag(aidPointHandle.getPosition)/getMag(peakPointHandle.getPosition), getHue(aidPointHandle.getPosition)-getHue(peakPointHandle.getPosition)];
    
    peakPointHandle.setColor(color);
    aidPointHandle.setColor(color);
    seedPointHandle.setColor([1 1 1]);
    
    kernel = exp(1i*theta).*sigmoid(a*hueDiff(theta/(2*pi) - getHue(peakPointHandle.getPosition)) + b);
    kernel(end+1) = nan + nan * 1i;
    kernel(end+1) = getMag(peakPointHandle.getPosition) * exp(1i*getHue(peakPointHandle.getPosition)*2*pi);
    kernel(end+1) = getMag(aidPointHandle.getPosition) * exp(1i*getHue(peakPointHandle.getPosition)*2*pi);
    kernel(end+1) = getMag(aidPointHandle.getPosition) * exp(1i*(getHue(peakPointHandle.getPosition) - kernelHandle.UserData(2))*2*pi);
    kernel(end+1) = getMag(aidPointHandle.getPosition) * exp(1i*getHue(peakPointHandle.getPosition)*2*pi);
    kernel(end+1) = getMag(aidPointHandle.getPosition) * exp(1i*(getHue(peakPointHandle.getPosition) + kernelHandle.UserData(2))*2*pi);
    kernel(end+1) = getMag(aidPointHandle.getPosition) * exp(1i*getHue(peakPointHandle.getPosition)*2*pi);
    kernelHandle.XData = real(kernel);
    kernelHandle.YData = imag(kernel);
    kernelHandle.Color = color;
    
    pixelScore = flatten(vals .^ (getMag(peakPointHandle.getPosition).^2) .* sigmoid( a*hueDiff(hues - getHue(peakPointHandle.getPosition)) + b ));
    
    seedPointPixelLabel = sum((round(getPosition(seedPointHandle))-[1 0]) .* [imageOriginalDims(2)*imageOriginalDims(3), 1]);
    
    selection = flood(seedPointPixelLabel, find(pixelScore > prctile(pixelScore, 100 * (kernelHandle.UserData(1)).^(0.30))), imageOriginalDims); % TODO
    
    contourCoreHandle.ZData(:) = 0;
    contourCoreHandle.ZData(selection) = contourCoreHandle.ZData(selection) + 1;
    contourCoreHandle.LevelList = 0.5;
    contourCoreHandle.LineColor = color;
    contourCoreHandle.LineWidth = 1;
    contourCoreHandle.Visible = 'on';
    
    
    % contourThinHandle.ZData = vals .^ (getMag(peakPointHandle.getPosition).^2) .* sigmoid( a*hueDiff(hues - getHue(peakPointHandle.getPosition)) + b );
    % contourThinHandle.LevelList = prctile(flatten(contourThinHandle.ZData), 100 * (kernelHandle.UserData(1)).^(0.05));
    % contourThinHandle.LineColor = color;
    % contourThinHandle.LineWidth = 1;
    % if(contourThinHandle.UserData)
    %     contourThinHandle.Visible = 'on';
    % else
    %     contourThinHandle.Visible = 'off';
    % end
    % 
    % seedPixels = find(contourThinHandle.ZData > contourThinHandle.LevelList);
    % groupSize = getSizeOfPixelPatch(seedPixels, imageOriginalDims);
    % selection = seedPixels(groupSize == max(groupSize));
    % 
    % contourCoreHandle.ZData = contourThinHandle.ZData;
    % contourCoreHandle.ZData(selection) = contourCoreHandle.ZData(selection) + 1;
    % contourCoreHandle.LevelList = 1 + contourThinHandle.LevelList;
    % contourCoreHandle.LineColor = color;
    % contourCoreHandle.LineWidth = 3;
    % contourCoreHandle.Visible = 'on';
    % 
    % % seedPixels = find(contourCoreHandle.ZData > contourCoreHandle.LevelList);
    % % contourThinHandle.ZData = vals + getConnectedPixels(seedPixels, vals > mean(vals(seedPixels))*kernelHandle.UserData(1), imageOriginalDims);
    % % contourThinHandle.LevelList = 1;
    % % contourThinHandle.LineColor = color;
    % % contourThinHandle.LineWidth = 2;
    % % contourThinHandle.Visible = 'off';
    
    function result = flood(seedPoint, pixelSet, imageDims)
    
    X = imageDims(1);
    Y = imageDims(2);
    Z = imageDims(3);
    
    getX = @(p) ceil(p/(Y*Z));
    getR = @(p) mod((p-1), (Y*Z)) + 1; % returns row number of pixel p
    getY = @(p) mod((getR(p)-1), Y) + 1;
    getZ = @(p) ceil(getR(p)/Y);
    
    result = [];
    
    queue = seedPoint;
    while(~isempty(queue))
        neighbors = [];
        
        neighbors = [neighbors; queue(getY(queue) > 1)-1];
        neighbors = [neighbors; queue(getY(queue) < Y)+1];
        neighbors = [neighbors; queue(getX(queue) > 1)-Y*Z];
        neighbors = [neighbors; queue(getX(queue) < X)+Y*Z];
        neighbors = [neighbors; queue(getZ(queue) > 1)-Y];
        neighbors = [neighbors; queue(getZ(queue) < Z)+Y];
        
        %         neighbors = unique(neighbors);
        [neighbors, j] = intersect(pixelSet, neighbors);
        pixelSet(j) = [];
        
        result = [result; neighbors];
        
        queue = neighbors;
    end
    
    
    
    function pixelGroupSize = getSizeOfPixelPatch(pixelSet, imageDims)
    
    X = imageDims(1);
    Y = imageDims(2);
    Z = imageDims(3);
    
    % getX = @(p) 1+floor((p-1)/(Y*Z));
    getX = @(p) ceil(p/(Y*Z));
    getR = @(p) mod((p-1), (Y*Z)) + 1; % returns row number of pixel p
    getY = @(p) mod((getR(p)-1), Y) + 1;
    % getZ = @(p) 1+floor((getR(p)-1)/Y);
    getZ = @(p) ceil(getR(p)/Y);
    
    label = zeros(size(pixelSet));
    numberOfPatches = 0;
    patchSize = [];
    
    for i = 1:numel(pixelSet)
        s = pixelSet(i);
        if(label(i) ~= 0)
            continue;
        end
        
        numberOfPatches = numberOfPatches + 1;
        label(i) = numberOfPatches;
        patchSize(numberOfPatches) = 1;
        
        queue = s;
        
        while(~isempty(queue))
            neighbors = [];
            
            neighbors = [neighbors; queue(getY(queue) > 1)-1];
            neighbors = [neighbors; queue(getY(queue) < Y)+1];
            neighbors = [neighbors; queue(getX(queue) > 1)-Y*Z];
            neighbors = [neighbors; queue(getX(queue) < X)+Y*Z];
            neighbors = [neighbors; queue(getZ(queue) > 1)-Y];
            neighbors = [neighbors; queue(getZ(queue) < Z)+Y];
            
            %         neighbors = unique(neighbors);
            [neighbors, j] = intersect(pixelSet, neighbors);
            neighbors(label(j) ~= 0) = [];
            
            label(j) = numberOfPatches;
            patchSize(numberOfPatches) = patchSize(numberOfPatches) + numel(neighbors);
            
            queue = neighbors;
        end
    end
    
    pixelGroupSize = zeros(size(pixelSet));
    
    for p = 1:numberOfPatches
        pixelGroupSize(label == p) = patchSize(p);
    end
    
        
    
    
    function resultMat = getConnectedPixels(seedPixels, validMat, imageDims)
    
    X = imageDims(1);
    Y = imageDims(2);
    Z = imageDims(3);
    
    % getX = @(p) 1+floor((p-1)/(Y*Z));
    getX = @(p) ceil(p/(Y*Z));
    getR = @(p) mod((p-1), (Y*Z)) + 1; % returns row number of pixel p
    getY = @(p) mod((getR(p)-1), Y) + 1;
    % getZ = @(p) 1+floor((getR(p)-1)/Y);
    getZ = @(p) ceil(getR(p)/Y);
    
    resultMat = zeros(size(validMat), 'logical');
    
    queue = seedPixels;
    resultMat(queue) = true;
    
    while(~isempty(queue))
        neighbors = [];
        
        neighbors = [neighbors; queue(getY(queue) > 1)-1];
        neighbors = [neighbors; queue(getY(queue) < Y)+1];
        neighbors = [neighbors; queue(getX(queue) > 1)-Y*Z];
        neighbors = [neighbors; queue(getX(queue) < X)+Y*Z];
        neighbors = [neighbors; queue(getZ(queue) > 1)-Y];
        neighbors = [neighbors; queue(getZ(queue) < Z)+Y];
        
        neighbors = unique(neighbors);
        neighbors(~validMat(neighbors)) = [];
        neighbors(resultMat(neighbors)) = [];
        
        queue = neighbors;
        resultMat(queue) = true;
    end
    
    
    
    % --- Executes on button press of accept button in rainbow plot.
    function rainbowButtonAccept_Callback(hObject, eventdata, original_hObjext, handles)
    % hObject    handle to buttonRainbow (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    flatten = @(x) x(:);
    
    mainFig = handles.mainFig;
    
    newROIs = zeros(size(mainFig.UserData.imageStack, 1), size(mainFig.UserData.imageStack, 2), numel(handles.rainbowContourCoreHandles));
    
    % pixelSTD = std(handles.imageStack, 0, 3);
    % pixelCOR = abs(handles.rainbowImageFFT);
    % pixelOptimalWeight = pixelCOR ./ (1 - pixelCOR.^2) ./ pixelSTD;
    
    for i = 1:numel(handles.rainbowContourCoreHandles)
        newROIs(:, :, i) = handles.rainbowContourCoreHandles{i}.ZData > handles.rainbowContourCoreHandles{i}.LevelList;
        
    %     newROIs(:, :, i) = newROIs(:, :, i) .* pixelOptimalWeight;
    %     newROIs(:, :, i) = newROIs(:, :, i) ./ max(flatten(newROIs(:, :, i)));
    end
    
    figure;
    for i = 1:size(newROIs, 3)
        subplot(ceil(sqrt(size(newROIs, 3))), ceil(sqrt(size(newROIs, 3))), i);
        imagesc(newROIs(:, :, i));
        colormap([0 0 0; jet(256)]);
        axis off;
        axis equal;
    end
    
    getHue = @(p) mod(atan2(p(2) , p(1)), 2*pi)/(2*pi);
    [~, order] = sort(arrayfun(@(x) -getHue(x{1}.getPosition), handles.peakPointHandles));
    
    
    newIdentities = mainFig.UserData.maxIdentityNum + (1:(size(newROIs, 3)+1));
    mainFig.UserData.maxIdentityNum = newIdentities(end);
    
    handles = modifyROIs(handles, newROIs(:, :, order), zeros(1, size(newROIs, 3)), newIdentities);
    refreshListBox(handles);
    handles.listboxROIs.Value = [];
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    
    % guidata(handles.mainFig, handles);
    guidata(original_hObjext, handles);
    
    % close(handles.rainbowFigure);
    
    
    
    % --- Executes on button press of accept button in rainbow plot.
    function rainbowButtonCancel_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonRainbow (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % % ================ replaced by the ski algorithm. Commenting out. ==============
    % function [tuningAngles, activityProfileOfROIs, tuningAngleFigureHandle, correlationMatrix] = estimateTuningAnglesOfROIs(handles, displayOption)
    % 
    %     if(nargin >= 2 && (strcmp(displayOption, 'on') || displayOption == true))
    %         tuningAngleFigureHandle = figure;
    %         title('Please Wait...');
    %         axis off;
    %         drawnow;
    %     else
    %         tuningAngleFigureHandle = [];
    %     end
    %     
    %     activityProfileOfROIs = zeros(size(handles.imageStack, 3), size(handles.ROIs, 3));
    %     
    %     for i_r = 1:size(handles.ROIs, 3)
    %         activityProfileOfROIs(:, i_r) = permute(mean(mean(double(handles.imageStack) .* handles.ROIs(:, :, i_r), 2), 1), [3 1 2]);
    %     end
    %     
    %     tuningAngles = linspace(0, 2*pi, size(handles.ROIs, 3)+1);
    %     tuningAngles = tuningAngles(1:end-1);
    %     tuningAngles = tuningAngles(randperm(numel(tuningAngles)));
    % 
    %     correlationMatrix = corrcoef(activityProfileOfROIs);
    % 
    %     if(~isempty(tuningAngleFigureHandle))
    %         clf(tuningAngleFigureHandle);
    %         figure(tuningAngleFigureHandle);
    %         tuningAngleFigureHandle.Color = [0 0 0];
    %         plot(exp(1i*linspace(0, 2*pi, 100)), 'w-');
    %         hold on;
    %         tuningAnglePlotHandle = plot(exp(1i*tuningAngles), 'o', 'MarkerSize', 10, 'Color', [0.9 0.2 0.2], 'LineWidth', 2);
    %         axis equal;
    %         axis off
    %         axis([-1 1 -1 1]);
    %     end
    %     
    %     while(true)
    %         
    %         [theta_i, theta_j] = meshgrid(tuningAngles, tuningAngles);
    %         gradients = sum(4 * (correlationMatrix - cos(theta_i - theta_j)) .* sin(theta_i - theta_j), 1);
    %         tuningAngles = tuningAngles - gradients * 0.001;
    %         
    %         if(~isempty(tuningAngleFigureHandle) && rand > 0.9)
    %             tuningAnglePlotHandle.XData = cos(tuningAngles);
    %             tuningAnglePlotHandle.YData = sin(tuningAngles);
    %             title({'Estimating Tuning Angles For Current ROIs', sprintf('(|G| = %0.5f)', sqrt(sum(gradients.^2)))}, 'Color', [1 1 1]);
    %             drawnow;
    %         end
    %         
    %         if(sqrt(sum(gradients.^2)) < 1e-4)
    %             break;
    %         end
    %     end
    %     
    %     [theta_i, theta_j] = meshgrid(tuningAngles, tuningAngles);
    %     finalError = sum(sum((correlationMatrix - cos(theta_i - theta_j)).^2)) ./ (numel(tuningAngles).^2 - tuningAngles);
    %     
    %     if(~isempty(tuningAngleFigureHandle))
    %         tuningAnglePlotHandle.XData = cos(tuningAngles);
    %         tuningAnglePlotHandle.YData = sin(tuningAngles);
    %         title(sprintf('Estimates for Tuning Angles of Current ROIs (E = %0.5f, |G| = %0.5f)',finalError,  sqrt(sum(gradients.^2))), 'Color', [1 1 1]);
    %         drawnow;
    %     end
    %   ==================================================================================
        
    % --- Executes on button press in buttonSortCircular.
    function buttonSortCircular_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonSortCircular (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    tuningAngles = angle(mainFig.UserData.tuningVectors);
    
    [~, reorder] = sort(mod(tuningAngles-tuningAngles(1), 2*pi));
    
    handles = modifyROIs(handles, mainFig.UserData.ROIs(:, :, reorder), mainFig.UserData.tuningVectors(reorder), mainFig.UserData.identityTracker(reorder));
    refreshListBox(handles);
    handles.listboxROIs.Value = [];
    
    refreshButtons(handles);
    handles = updateDisplayForROIs(handles);
    
    guidata(hObject, handles);
    % close(tuningAngleFigureHandle);
    
    
    % % --- Executes on button press in toggleOrderMode.
    % function toggleOrderMode_Callback(hObject, eventdata, handles)
    % % hObject    handle to toggleOrderMode (see GCBO)
    % % eventdata  reserved - to be defined in a future version of MATLAB
    % % handles    structure with handles and user data (see GUIDATA)
    % 
    % % Hint: get(hObject,'Value') returns toggle state of toggleOrderMode
    % 
    % modeStringList = {'Fixed', 'Dynamic'};
    % 
    % handles.rainbowOrderModeString = modeStringList{mod(find(strcmp(modeStringList, handles.rainbowOrderModeString)), numel(modeStringList))+1};
    % 
    % 
    % handles.toggleOrderMode.String = [handles.toggleOrderMode.String(1:strfind(handles.toggleOrderMode.String, ':')), ' ', handles.rainbowOrderModeString];
    % 
    % 
    % guidata(hObject, handles);
    % 
    % 
    % % --- Executes on button press in toggleAngleMode.
    % function toggleAngleMode_Callback(hObject, eventdata, handles)
    % % hObject    handle to toggleAngleMode (see GCBO)
    % % eventdata  reserved - to be defined in a future version of MATLAB
    % % handles    structure with handles and user data (see GUIDATA)
    % 
    % % Hint: get(hObject,'Value') returns toggle state of toggleAngleMode
    % 
    % modeStringList = {'Fixed', 'Flexible'};
    % 
    % handles.rainbowAngleModeString = modeStringList{mod(find(strcmp(modeStringList, handles.rainbowAngleModeString)), numel(modeStringList))+1};
    % 
    % 
    % 
    % handles.toggleAngleMode.String = [handles.toggleAngleMode.String(1:strfind(handles.toggleAngleMode.String, ':')), ' ', handles.rainbowAngleModeString];
    % 
    % guidata(hObject, handles);
    
    
    % --- Executes on button press in buttonSki.
    function buttonSki_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonSki (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    wb = waitbar(0, 'Calculating activity profiles...');
    
    % flatten = @(x) x(:);
    % smoothen = @(y, sigma) conv(y, normpdf(ceil(-4*sigma):floor(4*sigma), 0, sigma), 'same') ./ conv(ones(size(y)), normpdf(ceil(-4*sigma):floor(4*sigma), 0, sigma), 'same');
    % 
    % activityProfileOfROIs = zeros(size(handles.imageStack, 3), size(handles.ROIs, 3));
    % 
    % for i_r = 1:size(handles.ROIs, 3)
    %     activityProfileOfROIs(:, i_r) = permute(mean(mean(double(handles.imageStack) .* handles.ROIs(:, :, i_r), 2), 1), [3 1 2]);
    %     wb = waitbar(i_r/size(handles.ROIs, 3), wb, 'Calculating activity profiles...');
    % end
    
    mainFig = handles.mainFig;
    
    activityProfileOfROIs = calculateActivityProfiles(handles, wb);
    
    %     % ===== FAILED ATTEMPT AT IMPROVING THINGS BY FACTORING OUT THE BASELINE ACTIVITY OF DARKEST PIXELS ===== (keeping code anyway)
    %     activityProfileBaseline = zeros(size(handles.imageStack, 3), 2);
    %     avgImage = mean(double(handles.imageStack), 3);
    %
    %     baseROIs(:, :, 1) = avgImage < prctile(flatten(avgImage), 5);
    %     for sliceNumber = 1:handles.originalDims(4)
    %         sliceRows = (1:handles.originalDims(1)) + (sliceNumber-1)*handles.originalDims(1);
    %         baseROIs(sliceRows, :, 2) = avgImage(sliceRows, :) < prctile(flatten(avgImage(sliceRows, :)), 5);
    %     end
    %
    %     for i = 1:size(activityProfileBaseline, 2)
    %         activityProfileBaseline(:, i) = permute(mean(mean(double(handles.imageStack) .* baseROIs(:, :, i), 2), 1), [3 1 2]);
    %     end
    %
    %     residualActivity = zeros(size(activityProfileOfROIs));
    % %     residualActivityFromSmoothed = zeros(size(activityProfileOfROIs));
    %     for i = 1:size(activityProfileOfROIs, 2)
    %         [~, ~, residuals] = regress(activityProfileOfROIs(:, i), [ones(handles.originalDims(3), 1), activityProfileBaseline]);
    % %         [~, ~, residualsFromSmoothed] = regress(activityProfileOfROIs(:, i), [ones(handles.originalDims(3), 1), smoothen(activityProfileBaseline(:, 1), 10), smoothen(activityProfileBaseline(:, 2), 10)]);
    %         residualActivity(:, i) = residuals;
    % %         residualActivityFromSmoothed(:, i) = residualsFromSmoothed;
    %     end
    
    normalizedActivityProfileOfROIs = (activityProfileOfROIs - mean(activityProfileOfROIs, 1)) ./ std(activityProfileOfROIs, [], 1);
    normalizedActivityProfileOfROIs = normalizedActivityProfileOfROIs - mean(normalizedActivityProfileOfROIs, 2);
    
    %     correlationMatrix = corrcoef(activityProfileOfROIs);
    correlationMatrix = corrcoef(normalizedActivityProfileOfROIs); % I am not sure why this works better than the commented line above.
    
    kappa = str2double(handles.fieldKappa.String);
    if(isempty(kappa))
        kappa = 1.5;
        handles.fieldKappa.String = num2str(kappa);
    end
    
    close(wb);
    skiFigureHandle = figure;
    mainFig.UserData.tuningVectors = estimateVonMisesTunings(correlationMatrix, skiFigureHandle, kappa, 0);
    
    handles = updateDisplayForROIs(handles);
    
    guidata(hObject, handles);
    
    
    function fieldKappa_Callback(hObject, eventdata, handles)
    % hObject    handle to fieldKappa (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'String') returns contents of fieldKappa as text
    %        str2double(get(hObject,'String')) returns contents of fieldKappa as a double
    
    
    % --- Executes during object creation, after setting all properties.
    function fieldKappa_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to fieldKappa (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: edit controls usually have a white background on Windows.
    %       See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end
    
    
    % --- Executes on button press in buttonMirrorTunings.
    function buttonMirrorTunings_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonMirrorTunings (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    handles = modifyROIs(handles, mainFig.UserData.ROIs, -real(mainFig.UserData.tuningVectors) + 1i*imag(mainFig.UserData.tuningVectors), mainFig.UserData.identityTracker);
    
    handles = updateDisplayForROIs(handles);
    
    guidata(hObject, handles);
    
    
    function activityProfileOfROIs = calculateActivityProfiles(handles, wb)
    
    flatten = @(x) x(:);
    
    mainFig = handles.mainFig;
    
    if(exist('wb', 'var'))
        waitbar(0, wb);
    end
    
    activityProfileOfROIs = zeros(size(mainFig.UserData.imageStack, 3), size(mainFig.UserData.ROIs, 3));
    for i_r = 1:size(mainFig.UserData.ROIs, 3)
        identityOfThisROI = mainFig.UserData.identityTracker(i_r);
        if(numel(mainFig.UserData.activityProfiles) < identityOfThisROI || isempty(mainFig.UserData.activityProfiles{identityOfThisROI}))
            activityProfileOfROIs(:, i_r) = permute(mean(mean(double(mainFig.UserData.imageStack) .* mainFig.UserData.ROIs(:, :, i_r), 2), 1) ./ sum(flatten(mainFig.UserData.ROIs(:, :, i_r))), [3 1 2]);
            mainFig.UserData.activityProfiles{identityOfThisROI} = activityProfileOfROIs(:, i_r);
        else
            activityProfileOfROIs(:, i_r) = mainFig.UserData.activityProfiles{identityOfThisROI};
        end
        if(exist('wb', 'var'))
            waitbar(i_r/size(mainFig.UserData.ROIs, 3), wb);
        end
    end
    
    % --- Executes on button press in buttonExtractPhase.
    function buttonExtractPhase_Callback(hObject, eventdata, handles)
    % hObject    handle to buttonExtractPhase (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    mainFig = handles.mainFig;
    
    if(isempty(mainFig.UserData.timestamps))
        tmp = warndlg('You must provide a file that contains the timestamps for current dataset.', 'Timestamps Not Found');
        uiwait(tmp);
        [filename,path] = uigetfile('*.mat');
        load([path, filename], 'timestamps');
        assert(size(timestamps, 3) == mainFig.UserData.originalDims(3) && size(timestamps, 4) == mainFig.UserData.originalDims(4));
        flatten = @(x) x(:);
        mainFig.UserData.timestamps = flatten(mean(timestamps, 4));
    end
    
    wb = waitbar(0, 'Calculating activity profiles...');
    F = calculateActivityProfiles(handles, wb);
    
    
    % ignoreTheseFrames = (mainFig.UserData.timestamps == 0);
    % if(any(ignoreTheseFrames))
    %     warning('%d of the Unix timestamps are 0. Ignoring those frames', sum(ignoreTheseFrames));
    % end
    
    modernEra = 1.7e9; % assuming the recordings were taken in the modern era (unix time in seconds).
    ignoreTheseFrames = abs(mainFig.UserData.timestamps/1.7e9 - 1) > 0.25;
    
    if(all(ignoreTheseFrames) && any(abs(mainFig.UserData.timestamps/(modernEra*1e9) - 1) < 0.25))
        warning('converting unix time from nanoseconds!');
        mainFig.UserData.timestamps = double(mainFig.UserData.timestamps)/1e9;
        ignoreTheseFrames = abs(mainFig.UserData.timestamps/modernEra - 1) > 0.25;
    end
    
    assert(~all(ignoreTheseFrames));
    if(any(ignoreTheseFrames))
        warning('%d of the Unix timestamps corrupted. Ignoring those frames', sum(ignoreTheseFrames));
    end
    
    F = F(~ignoreTheseFrames, :);
    volumeTimestamps = mainFig.UserData.timestamps(~ignoreTheseFrames);
    
    deltaFoverF = (F - prctile(F, 5))./prctile(F, 5);
    
    
    if(handles.useTuningCheckbox.Value)
        tuningAngle = angle(mainFig.UserData.tuningVectors);
    else
        tuningAngle = linspace(0, 2*pi, 16+1);
        tuningAngle(end) = [];
    end
    
    
    phaseDecoded = nan(size(F, 1), 2);
    
    kappa = str2double(handles.fieldKappa.String);
    if(isempty(kappa))
        kappa = 1.75;
        handles.fieldKappa.String = num2str(kappa);
    end
    
    vonMises = @(x, mu, kappa) exp(kappa .* cos(x - mu)) ./ (2*pi*besseli(0, kappa)); % circular gaussian function
    for t = 1:size(F, 1)
        waitbar((t-1)/size(F, 1), wb, 'Extracting phase...');
        
        phaseDecoded(t, 1) = atan2(nansum(sin(tuningAngle).*deltaFoverF(t, :)), nansum(cos(tuningAngle).*deltaFoverF(t, :)));
        phaseDecoded(t, 2) = fminbnd(@(x) sqrt(nansum((deltaFoverF(t, :) - vonMises(tuningAngle, x(1), kappa)).^2)), 0, 4*pi);
    
    end
    
    
    lowPassFilterPhase = phaseDecoded;
    
    tau = str2double(handles.fieldTau.String);
    if(isempty(kappa))
        tau = 0.2;
        handles.fieldTau.String = num2str(tau);
    end
    
    for i = 2:size(phaseDecoded, 1)
        delta = mod(phaseDecoded(i, :) - lowPassFilterPhase(i-1, :)+pi, 2*pi) -pi;
        lowPassFilterPhase(i, :) = lowPassFilterPhase(i-1, :) + (1-exp((volumeTimestamps(i-1)-volumeTimestamps(i))/tau))*delta;
        waitbar((i-1)/size(phaseDecoded, 1), wb, 'Low pass filtering...');
    end
    
    
    finalPhaseEstimate = mod(mean(unwrap(lowPassFilterPhase')), 2*pi)';
    
    figureHandle = figure('Position', [400, 1200, 1800, 240] + [1 1 0 0].*randn(1, 4)*40);
    axisHandle = axes('Position', [0.05, 0.2, 0.8, 0.7]);
    
    timeScale = 1/60;
    timeScaleString = 'min';
    
    imagesc([0, range(volumeTimestamps)]*timeScale, [0 360], deltaFoverF', 'Parent', axisHandle);
    colormap(circshift(hot(64), 1, 2));
    hold on;
    plot((volumeTimestamps-volumeTimestamps(1))*timeScale,  mod(finalPhaseEstimate', 2*pi)*180/pi, 'y.', 'Parent', axisHandle);
    xlabel(sprintf('time (%s) from %s', timeScaleString, datestr(datetime(volumeTimestamps(1),'ConvertFrom','epochtime','TicksPerSecond',1,'Format','dd MMM yyyy, HH:mm:ss.SSS'))));
    set(gca, 'YDir', 'normal');
    yticks(gca, [0 90 180 270 360]);
    ylabel('phase (deg)');
    
    D = struct;
    
    D.ROIs = mainFig.UserData.ROIs;
    D.tuningVectors = mainFig.UserData.tuningVectors;
    
    D.activityProfiles = F;
    D.deltaFoverF = deltaFoverF;
    D.tuningAngle = tuningAngle;
    D.phaseDecoded = phaseDecoded;
    D.lowPassFilterPhase = lowPassFilterPhase;
    D.kappa = kappa;
    D.tau = tau;
    D.finalPhaseEstimate = finalPhaseEstimate;
    D.timestamps = volumeTimestamps;
    
    saveButton = uicontrol(...
        'Parent', figureHandle, ...
        'Style', 'pushbutton', ...
        'String', 'Export Data', ...
        'Units', 'normalized', ...
        'Position', [0.90 0.20 0.05 0.1], ...
        'Visible', 'on', ...
        'Callback', @(h, e) exportDataUI(D, strrep(mainFig.UserData.fileName, '.mat', '.phaseData.mat')));
    
    close(wb);
    
    % refreshListBox(handles);
    % refreshButtons(handles);
    % handles = updateDisplayForROIs(handles);
    % 
    guidata(hObject, handles);
    
    function exportDataUI(data, defaultFilename)
    
        if(~exist('defaultFilename', 'var'))
            defaultFilename = 'untitled.mat';
        end
    
        [file, path] = uiputfile(defaultFilename, 'Export Phase Data');
        save([path, file], '-struct', 'data');
    
    % --- Executes on button press in useTuningCheckbox.
    function useTuningCheckbox_Callback(hObject, eventdata, handles)
    % hObject    handle to useTuningCheckbox (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hint: get(hObject,'Value') returns toggle state of useTuningCheckbox
    
    if(handles.useTuningCheckbox.Value)
        tmp = warndlg('The extracted tuning angles may be biased, especially if the circular space was not sampled uniformly during the imaging session. Uncheck box to assume equally spaced tuning angles in the same order as the ROI list. (Sort the list if needed. You can use empty ROIs as placeholders.)', 'Not recommended');
        uiwait(tmp);
    end
    
    
    function fieldTau_Callback(hObject, eventdata, handles)
    % hObject    handle to fieldTau (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    structure with handles and user data (see GUIDATA)
    
    % Hints: get(hObject,'String') returns contents of fieldTau as text
    %        str2double(get(hObject,'String')) returns contents of fieldTau as a double
    
    
    % --- Executes during object creation, after setting all properties.
    function fieldTau_CreateFcn(hObject, eventdata, handles)
    % hObject    handle to fieldTau (see GCBO)
    % eventdata  reserved - to be defined in a future version of MATLAB
    % handles    empty - handles not created until after all CreateFcns called
    
    % Hint: edit controls usually have a white background on Windows.
    %       See ISPC and COMPUTER.
    if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
        set(hObject,'BackgroundColor','white');
    end