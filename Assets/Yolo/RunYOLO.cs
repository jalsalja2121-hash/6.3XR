using System;
using System.Collections;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;
using UnityEngine.Android;

public class RunYOLO : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset modelAsset;
    public TextAsset classesAsset;

    [Header("UI Settings")]
    public RawImage displayImage;
    public Texture2D borderTexture;
    public Font font;

    [Header("Layout Settings")]
    [SerializeField, Range(0.4f, 0.8f)]
    private float cameraAreaRatio = 0.6f; // 왼쪽 카메라 영역 비율

    [Header("Detection Thresholds")]
    [SerializeField, Range(0, 1)] float iouThreshold = 0.5f;
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.5f;

    // GeminiManager에서 가져갈 마지막 감지 라벨
    public string lastDetectedLabel = "";

    private bool isProcessing = false;
    private int frameCount = 0;

    private const BackendType backend = BackendType.GPUCompute;

    private Worker worker;
    private string[] labels;

    private RenderTexture displayRT;
    private RenderTexture yoloRT;

    private Sprite borderSprite;
    private Transform displayLocation;

    private WebCamTexture webCamTexture;
    private List<GameObject> boxPool = new List<GameObject>();

    // YOLO 입력 크기
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    // 화면 표시용 와이드 해상도
    private const int displayWidthRT = 1280;
    private const int displayHeightRT = 720;

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
        Application.targetFrameRate = 60;

        // 휴대폰 가로 화면 고정
        Screen.autorotateToPortrait = false;
        Screen.autorotateToPortraitUpsideDown = false;
        Screen.autorotateToLandscapeLeft = true;
        Screen.autorotateToLandscapeRight = true;
        Screen.orientation = ScreenOrientation.LandscapeLeft;

        if (displayImage == null)
        {
            Debug.LogError("Display Image가 연결되지 않았습니다.");
            return;
        }

        SetupLeftCameraArea();

        if (classesAsset != null)
        {
            labels = classesAsset.text.Split('\n');
        }

        if (!Permission.HasUserAuthorizedPermission(Permission.Camera))
        {
            Permission.RequestUserPermission(Permission.Camera);
        }

        LoadModel();

        // 화면 표시용은 와이드
        displayRT = new RenderTexture(displayWidthRT, displayHeightRT, 0);
        displayRT.Create();

        // YOLO 분석용은 640x640
        yoloRT = new RenderTexture(imageWidth, imageHeight, 0);
        yoloRT.Create();

        displayImage.texture = displayRT;
        displayLocation = displayImage.transform;

        if (borderTexture != null)
        {
            borderSprite = Sprite.Create(
                borderTexture,
                new Rect(0, 0, borderTexture.width, borderTexture.height),
                new Vector2(0.5f, 0.5f)
            );
        }

        StartCoroutine(SetupInputCoroutine());
    }

    void SetupLeftCameraArea()
    {
        RectTransform rt = displayImage.rectTransform;

        // 왼쪽 영역만 YOLO 카메라 화면으로 사용
        // X 0 ~ 0.6 = 왼쪽 60%
        // X 0.6 ~ 1 = 오른쪽 답변 UI 영역
        rt.anchorMin = new Vector2(0f, 0f);
        rt.anchorMax = new Vector2(cameraAreaRatio, 1f);

        rt.offsetMin = Vector2.zero;
        rt.offsetMax = Vector2.zero;

        rt.pivot = new Vector2(0.5f, 0.5f);
        rt.localScale = Vector3.one;
        rt.localRotation = Quaternion.identity;
        rt.localEulerAngles = Vector3.zero;

        displayImage.color = Color.white;
        displayImage.raycastTarget = false;

        // 카메라 화면을 가장 뒤로 보냄
        displayImage.transform.SetAsFirstSibling();
    }

    void LoadModel()
    {
        if (modelAsset == null)
        {
            Debug.LogError("YOLO modelAsset이 연결되지 않았습니다.");
            return;
        }

        var model1 = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(
            new TensorShape(4, 4),
            new float[]
            {
                1,      0,      1,      0,
                0,      1,      0,      1,
                -0.5f,  0,      0.5f,   0,
                0,      -0.5f,  0,      0.5f
            }
        );

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

    IEnumerator SetupInputCoroutine()
    {
        yield return new WaitUntil(() => Permission.HasUserAuthorizedPermission(Permission.Camera));

        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length > 0)
        {
            webCamTexture = new WebCamTexture(devices[0].name, 1280, 720, 30);
            webCamTexture.Play();

            Debug.Log("카메라가 시작되었습니다: " + devices[0].name);

            yield return new WaitUntil(() => webCamTexture.width > 100);

            Debug.Log("카메라 해상도: " + webCamTexture.width + " x " + webCamTexture.height);
        }
        else
        {
            Debug.LogError("사용 가능한 카메라가 없습니다.");
        }
    }

    private void Update()
    {
        if (webCamTexture != null &&
            webCamTexture.isPlaying &&
            webCamTexture.didUpdateThisFrame &&
            webCamTexture.width > 100)
        {
            // RawImage 자체 회전 금지
            // 회전시키면 왼쪽 영역 배치와 박스 좌표가 틀어질 수 있음
            displayImage.rectTransform.localRotation = Quaternion.identity;
            displayImage.rectTransform.localEulerAngles = Vector3.zero;

            // 상하 반전만 처리
            if (webCamTexture.videoVerticallyMirrored)
            {
                displayImage.uvRect = new Rect(0, 1, 1, -1);
            }
            else
            {
                displayImage.uvRect = new Rect(0, 0, 1, 1);
            }

            // 화면 표시용: 왼쪽 카메라 영역에 와이드 화면 표시
            Graphics.Blit(webCamTexture, displayRT);
            displayImage.texture = displayRT;

            // YOLO 분석용: 640x640으로 분석
            Graphics.Blit(webCamTexture, yoloRT);

            _ = ExecuteML();
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
    }

    public async Task ExecuteML()
    {
        frameCount++;

        // 5프레임마다 한 번만 추론
        if (frameCount % 5 != 0)
            return;

        if (isProcessing)
            return;

        isProcessing = true;

        Tensor<float> output = null;
        Tensor<int> labelIDs = null;

        try
        {
            if (webCamTexture == null ||
                !webCamTexture.isPlaying ||
                webCamTexture.width < 100 ||
                worker == null ||
                yoloRT == null)
            {
                return;
            }

            using Tensor<float> inputTensor = new Tensor<float>(
                new TensorShape(1, 3, imageHeight, imageWidth)
            );

            TextureConverter.ToTensor(yoloRT, inputTensor, default);

            worker.Schedule(inputTensor);

            var outputTensor = worker.PeekOutput("output_0") as Tensor<float>;
            var labelIDsTensor = worker.PeekOutput("output_1") as Tensor<int>;

            if (outputTensor == null || labelIDsTensor == null)
            {
                Debug.LogWarning("YOLO 출력 텐서를 읽지 못했습니다.");
                return;
            }

            output = await outputTensor.ReadbackAndCloneAsync();
            labelIDs = await labelIDsTensor.ReadbackAndCloneAsync();

            float displayWidth = displayImage.rectTransform.rect.width;
            float displayHeight = displayImage.rectTransform.rect.height;

            if (displayWidth <= 0 || displayHeight <= 0)
            {
                Debug.LogWarning("Display Image 크기가 0입니다.");
                return;
            }

            float scaleX = displayWidth / imageWidth;
            float scaleY = displayHeight / imageHeight;

            int boxesFound = output.shape[0];

            ClearAnnotations();
            lastDetectedLabel = "";

            for (int n = 0; n < Mathf.Min(boxesFound, 50); n++)
            {
                string detectedLabel = "Object";

                int labelIndex = labelIDs[n];

                if (labels != null && labelIndex >= 0 && labelIndex < labels.Length)
                {
                    detectedLabel = labels[labelIndex].Trim();
                }

                BoundingBox box = new BoundingBox
                {
                    centerX = output[n, 0] * scaleX - displayWidth / 2f,
                    centerY = output[n, 1] * scaleY - displayHeight / 2f,
                    width = output[n, 2] * scaleX,
                    height = output[n, 3] * scaleY,
                    label = detectedLabel
                };

                if (n == 0)
                {
                    lastDetectedLabel = box.label;
                }

                DrawBox(box, n, Mathf.Max(20f, displayHeight * 0.04f));
            }
        }
        catch (Exception e)
        {
            Debug.LogError("YOLO 실행 중 오류: " + e.Message);
        }
        finally
        {
            output?.Dispose();
            labelIDs?.Dispose();
            isProcessing = false;
        }
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

        RectTransform rt = panel.GetComponent<RectTransform>();

        rt.anchorMin = new Vector2(0.5f, 0.5f);
        rt.anchorMax = new Vector2(0.5f, 0.5f);
        rt.pivot = new Vector2(0.5f, 0.5f);

        rt.anchoredPosition = new Vector2(box.centerX, -box.centerY);
        rt.sizeDelta = new Vector2(box.width, box.height);

        Text label = panel.GetComponentInChildren<Text>();

        if (label != null)
        {
            label.text = box.label;
            label.fontSize = Mathf.RoundToInt(fontSize);
        }
    }

    public GameObject CreateNewBox(Color color)
    {
        GameObject panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();

        Image img = panel.AddComponent<Image>();
        img.color = new Color(color.r, color.g, color.b, 0.35f);

        if (borderSprite != null)
        {
            img.sprite = borderSprite;
            img.type = Image.Type.Sliced;
        }

        panel.transform.SetParent(displayLocation, false);

        RectTransform panelRect = panel.GetComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0.5f, 0.5f);
        panelRect.anchorMax = new Vector2(0.5f, 0.5f);
        panelRect.pivot = new Vector2(0.5f, 0.5f);
        panelRect.anchoredPosition = Vector2.zero;

        GameObject textObj = new GameObject("ObjectLabel");
        textObj.AddComponent<CanvasRenderer>();
        textObj.transform.SetParent(panel.transform, false);

        Text txt = textObj.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 30;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;
        txt.verticalOverflow = VerticalWrapMode.Overflow;

        RectTransform textRect = textObj.GetComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0, 1);
        textRect.anchorMax = new Vector2(0, 1);
        textRect.pivot = new Vector2(0, 1);
        textRect.anchoredPosition = new Vector2(5, -5);
        textRect.sizeDelta = new Vector2(300, 60);

        boxPool.Add(panel);

        return panel;
    }

    public void ClearAnnotations()
    {
        foreach (GameObject box in boxPool)
        {
            if (box != null)
            {
                box.SetActive(false);
            }
        }
    }

    void OnDestroy()
    {
        centersToCorners?.Dispose();
        worker?.Dispose();

        if (displayRT != null)
        {
            displayRT.Release();
            Destroy(displayRT);
        }

        if (yoloRT != null)
        {
            yoloRT.Release();
            Destroy(yoloRT);
        }

        if (webCamTexture != null)
        {
            webCamTexture.Stop();
        }
    }
}