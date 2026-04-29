using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;
using UnityEngine.Android; // 안드로이드 권한 요청을 위해 필수

public class RunYOLO : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset modelAsset;
    public TextAsset classesAsset;
    private bool isProcessing = false;

    [Header("UI Settings")]
    public RawImage displayImage;
    public Texture2D borderTexture;
    public Font font;

    // --- 최적화를 위한 변수들 ---
    private int frameCount = 0;
    const BackendType backend = BackendType.GPUCompute;
    private Transform displayLocation;
    private Worker worker;
    private string[] labels;
    private RenderTexture targetRT;
    private RenderTexture yoloRT;
    private Sprite borderSprite;

    private const int imageWidth = 640;
    private const int imageHeight = 640;
    private WebCamTexture webCamTexture;
    private List<GameObject> boxPool = new List<GameObject>();

    [Header("Detection Thresholds")]
    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    private Tensor<float> centersToCorners;

    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public string label;
    }

    void Start()
    {
        // 최신 기기 성능 대응 및 화면 고정
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;

        if (classesAsset != null)
            labels = classesAsset.text.Split('\n');

        // 1. 안드로이드 카메라 권한 확인 및 요청
        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
        }

        LoadModel();

        targetRT = new RenderTexture(imageWidth, imageHeight, 0);
        yoloRT = new RenderTexture(imageWidth, imageHeight, 0);
        displayLocation = displayImage.transform;

        // 2. 카메라 시작 (코루틴 방식으로 권한 획득 대기)
        StartCoroutine(SetupInputCoroutine());

        if (borderTexture != null)
        {
            borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), new Vector2(borderTexture.width / 2, borderTexture.height / 2));
        }
    }

    void LoadModel()
    {
        var model1 = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
            1,      0,      1,      0,
            0,      1,      0,      1,
            -0.5f,  0,      0.5f,   0,
            0,      -0.5f,  0,      0.5f
        });

        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model1);
        var modelOutput = Functional.Forward(model1, inputs)[0];
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);
        var allScores = modelOutput[0, 4.., ..];
        var scores = Functional.ReduceMax(allScores, 0);
        var classIDs = Functional.ArgMax(allScores, 0);
        var boxCorners = Functional.MatMul(boxCoords, Functional.Constant(centersToCorners));
        var indices = Functional.NMS(boxCorners, scores, iouThreshold, scoreThreshold);
        var coords = Functional.IndexSelect(boxCoords, 0, indices);
        var labelIDs = Functional.IndexSelect(classIDs, 0, indices);

        worker = new Worker(graph.Compile(coords, labelIDs), backend);
    }

    // 카메라 권한 승인 시까지 기다렸다가 실행
    IEnumerator SetupInputCoroutine()
    {
        yield return new WaitUntil(() => Permission.HasUserAuthorizedPermission(Permission.Camera));

        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            webCamTexture = new WebCamTexture(devices[0].name, 1280, 720);
            webCamTexture.Play();
            Debug.Log("카메라가 시작되었습니다: " + devices[0].name);
        }
        else
        {
            Debug.LogError("사용 가능한 카메라가 없습니다.");
        }
    }

    private void Update()
    {
        // 3. 카메라 데이터가 실제로 들어왔는지 확인 (검은 화면 방지 핵심)
        if (webCamTexture != null && webCamTexture.didUpdateThisFrame && webCamTexture.width > 100)
        {
            // 모바일 카메라 텍스처 회전/반전 보정
            float rotation = -webCamTexture.videoRotationAngle;
            displayImage.rectTransform.localEulerAngles = new Vector3(0, 0, rotation);

            // 상하 반전 대응 (안드로이드에서 자주 발생)
            if (webCamTexture.videoVerticallyMirrored)
                displayImage.uvRect = new Rect(0, 1, 1, -1);
            else
                displayImage.uvRect = new Rect(0, 0, 1, 1);

            Graphics.Blit(webCamTexture, targetRT);
            displayImage.texture = targetRT;

            // 4. 추론 실행
            _ = ExecuteML();
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
    }

    public async Task ExecuteML()
    {
        // 프레임 스킵 (최신 기기이므로 5~8 프레임 적절)
        frameCount++;
        if (frameCount % 5 != 0) return;

        if (isProcessing) return;
        isProcessing = true;

        if (webCamTexture == null || !webCamTexture.isPlaying || webCamTexture.width < 100)
        {
            isProcessing = false;
            return;
        }

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));

        Graphics.Blit(targetRT, yoloRT);
        TextureConverter.ToTensor(yoloRT, inputTensor, default);

        worker.Schedule(inputTensor);

        var outputTensor = worker.PeekOutput("output_0") as Tensor<float>;
        var labelIDsTensor = worker.PeekOutput("output_1") as Tensor<int>;

        // 비동기 데이터 읽기
        var output = await outputTensor.ReadbackAndCloneAsync();
        var labelIDs = await labelIDsTensor.ReadbackAndCloneAsync();

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;

        int boxesFound = output.shape[0];

        ClearAnnotations();

        for (int n = 0; n < Mathf.Min(boxesFound, 50); n++) // 최대 검출 개수 제한
        {
            var box = new BoundingBox
            {
                centerX = output[n, 0] * scaleX - displayWidth / 2,
                centerY = output[n, 1] * scaleY - displayHeight / 2,
                width = output[n, 2] * scaleX,
                height = output[n, 3] * scaleY,
                label = (labels != null && labelIDs[n] < labels.Length) ? labels[labelIDs[n]] : "Object",
            };
            DrawBox(box, n, displayHeight * 0.05f);
        }

        output.Dispose();
        labelIDs.Dispose();
        isProcessing = false;
    }

    public void DrawBox(BoundingBox box, int id, float fontSize)
    {
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
        }
        else
        {
            panel = CreateNewBox(Color.yellow);
        }

        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);
        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);

        var label = panel.GetComponentInChildren<Text>();
        if (label != null)
        {
            label.text = box.label;
            label.fontSize = (int)fontSize;
        }
    }

    public GameObject CreateNewBox(Color color)
    {
        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        Image img = panel.AddComponent<Image>();
        img.color = color;
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;
        panel.transform.SetParent(displayLocation, false);

        var textObj = new GameObject("ObjectLabel");
        textObj.AddComponent<CanvasRenderer>();
        textObj.transform.SetParent(panel.transform, false);
        Text txt = textObj.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = textObj.GetComponent<RectTransform>();
        rt2.anchorMin = new Vector2(0, 1); // 상단 좌측 기준
        rt2.anchorMax = new Vector2(0, 1);
        rt2.pivot = new Vector2(0, 0);
        rt2.offsetMin = new Vector2(5, 5);
        rt2.sizeDelta = new Vector2(200, 50);

        boxPool.Add(panel);
        return panel;
    }

    public void ClearAnnotations()
    {
        foreach (var box in boxPool)
        {
            if (box != null) box.SetActive(false);
        }
    }

    void OnDestroy()
    {
        centersToCorners?.Dispose();
        worker?.Dispose();
        if (targetRT != null) targetRT.Release();
        if (yoloRT != null) yoloRT.Release();
        if (webCamTexture != null) webCamTexture.Stop();
    }
}