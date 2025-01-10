const { createApp } = Vue;
const { createVuetify } = Vuetify;

let notifierGlobalOptions = {
  position: "bottom-right",
  icons: { enabled: false },
  minDurations: {
    async: 30,
    "async-block": 30,
  },
};

function _joinWords(words, conjunction = "and") {
  const names = words.map((item) => `"${item}"`);
  if (names.length == 1) {
    return names[0];
  }
  const last_name = names.pop();
  return `${names.join(", ")} ${conjunction} ${last_name}`;
}

function _joinTiers(tiers) {
  const unique_tiers = [];
  for (let tier of UPGRADABLE_TIERS) {
    if (tiers.includes(tier)) {
      unique_tiers.push(tier);
    }
  }
  return _joinWords(unique_tiers, "or");
}

var notifier = new AWN(notifierGlobalOptions);

const defaultTheme = {
  dark: true,
  colors: {
    background: "#0b0f19",
    primary: "#5a3cff",
  },
};

const vuetify = createVuetify({
  theme: {
    defaultTheme: "defaultTheme",
    themes: {
      defaultTheme,
    },
  },
});

createApp({
  delimiters: ["[[", "]]"],
  data() {
    return {
      settingTab: "settings",
      prompt: "",
      _performance: "Speed",
      performanceOptions: [],
      forcedSteps: null,
      aspectRatio: "1:1",
      aspectRatios: [],
      aspectRatiosNumber: [],
      imageNumber: 2,
      negativePrompt: "",
      isRandom: true,
      randomSeed: 0,
      selectedStyles: [],
      styles: [],
      baseModel: "",
      baseModels: [],
      refiner: "",
      refiners: [],
      refinerSwitch: 0.5,
      loraModels: [],
      numLoras: 0,
      selectedLoraModels: [],
      selectedLoraWeights: [],
      guidanceScale: 4.0,
      imageSharpness: 2.0,
      clipSkip: 1,
      defaultOptions: {},
      runningTaskId: "",
      runningTaskStatus: "",
      runningTaskProgress: 0,
      runningTaskMessage: "",
      runningTaskReturnUrl: false,
      runningTaskPreviewImage: "",
      runningTaskResultImages: [],
      runningTaskQueueLength: 0,
      runningTaskQueuePosiiton: 0,
      generating: false,
      generatingParams: {},
      waiting: true,
      maxHistory: 10,
      historyResults: [],
      previewDialog: false,
      previewIndex: 0,
      previewImages: [],
      likeButtonColor: {},
      favoriteButtonColor: {},
      shareButtonColor: {},
      skipLoading: false,
      stopLoading: false,
      panelName: "image_options",
      showImageOptions: false,
      imageOptionTab: "uov",
      uovSelection: "Disabled",
      uovMethods: [],
      imagePrompts: [],
      ipTypes: [],
      _numImagePrompts: 4,
      imagePromptImages: [],
      imagePromptAdvancedPanel: "",
      imagePromptDefaultValues: {},
      contentTypeSelection: "Photograph",
      contentTypes: [],
      describeImageUploader: null,
      describeErrorMessage: null,
      inpaintImageUploader: null,
      inpaintImageFiles: [],
      inpaintSelection: "Inpaint or Outpaint (default)",
      outpaintDirection: [],
      inpaintImprovementPrompt: "",
      inpaintModificationPrompt: "",
      inpaintStep: "Upload Image",
      uovImageFile: "",
      imageRules: [
        (value) => {
          return (
            !value ||
            !value.length ||
            value[0].size < 10000000 ||
            "Avatar size should be less than 10 MB!"
          );
        },
      ],
      imageEditor: null,
      imageOptionsCloseButtonHover: false,
      toolObserver: null,
      optionsObserver: null,
      describeImageLoading: false,
      selectedPreset: "",
      availablePresets: [],
      selectingPreset: false,
      loadingPreset: false,
      presetName: {
        default: "cinematic",
        sai: "stability ai",
      },
      presetColor: {
        default: "green",
        anime: "deep-orange",
        lcm: "light-green",
        realistic: "light-blue",
        sai: "pink",
        turbo: "amber",
        lighting: "blue-accent-3",
        sd3: "#00ff84",
        flux: "#f5ce62",
      },
      selectedSampler: "dpmpp_2m_sde_gpu",
      availableSamplers: [],
      selectedScheduler: "karras",
      availableSchedulers: [],
      styleOverlay: {
        show: false,
        src: "",
      },
      nonLCMArguments: {},
      estimateConsume: {
        args: {},
        timeoutId: null,
        inference: "-",
        discount: 0,
        imageNumber: 2,
      },
      estimateBlipConsume: {
        inference: "-",
        discount: 0,
      },
      estimateGptVisionConsume: {
        inference: "-",
        discount: 0,
      },
      gptVisionTask: {
        task_id: null,
        queueLength: 0,
        queuePosition: 0,
        result: null,
      },
      popup: {
        isOpen: false,
        itemName: "",
        title: "",
        message: "",
        confirmText: "",
        url: "",
      },
      safetyPopup: {
        isOpen: false,
        checkboxA: false,
        checkboxB: false,
        isAgreed: false,
      },
      userOrderInformation: null,
      _featurePermissions: null,
      taskType: "sdxl",
      flux: {
        background: "background: linear-gradient(to right, #f5ce62, #e43603);",
        textColor: "#ffffff",
      },
      sd3: {
        enabled: false,
        baseModel: "",
        baseModels: [],
        aspectRatio: "1:1",
        aspectRatios: [],
        strength: 0.5,
        estimateConsume: {
          inference: "-",
          discount: 0,
          imageNumber: 2,
        },
        background:
          "background: linear-gradient(270deg, rgb(0, 255, 239) 0%, rgb(0, 255, 132) 100%)",
        textColor: "#0c5536",
      },
    };
  },
  methods: {
    getGenerateButtonColor() {
      if (this.taskType === "sd3") {
        return this.sd3.background;
      } else if (this.taskType === "flux") {
        return this.flux.background;
      } else {
        return "";
      }
    },
    getGenerateButtonTextColor() {
      if (this.taskType === "sd3") {
        return this.sd3.textColor;
      } else if (this.taskType === "flux") {
        return this.flux.textColor;
      } else {
        return "#5a3cff";
      }
    },
    startIntroJS(force = false) {
      const cookie_key = "_fooocus_introjs_showed";
      if (!force && window.Cookies.get(cookie_key)) {
        return;
      }
      const introjs = getIntroJS();
      introjs.onexit(() =>
        window.Cookies.set(cookie_key, true, { expires: 360 }),
      );
      introjs.start();
    },
    async checkNSFW(is_nsfw) {
      if (is_nsfw.length === 0 || !is_nsfw[is_nsfw.length - 1]) {
        return;
      }
      await this.upgradePopup("NSFW_CONTENT");
    },
    async getSafetyAgreement() {
      const params = new URLSearchParams({ field: SAFETY_AGREEMENT_KEY });
      const url = `/api/user_profile?${params.toString()}`;

      const response = await fetch(url, {
        method: "GET",
        credentials: "include",
      });

      if (!response.ok) {
        throw `Get safety agreement agreed failed: ${response.statusText}`;
      }
      const content = await response.json();

      return content[SAFETY_AGREEMENT_KEY];
    },
    async setSafetyAgreement() {
      const url = "/api/user_profile";
      const body = {};
      body[SAFETY_AGREEMENT_KEY] = true;

      const response = await fetch(url, {
        method: "PATCH",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw `Set safety agreement agreed failed: ${response.statusText}`;
      }
      return await response.json();
    },
    async getDirtyWords() {
      if (!this._dirtyWords) {
        const response = await fetch("/public/dirty_words.txt", {
          method: "GET",
          credentials: "include",
        });
        if (!response.ok) {
          throw `Request dirty words failed: ${response.statusText}`;
        }
        const body = await response.text();

        const lines = body.toLowerCase().split(/[\r\n]+/);

        this._dirtyWords = [...new Set(lines)];
      }
      return this._dirtyWords;
    },
    async checkPromptDirtyWords(prompt) {
      prompt = prompt.toLowerCase();

      const words_1 = prompt.split(/\W+/).filter(Boolean);
      const words_2 = prompt.split(/[\s,]+/).filter(Boolean);

      const words = new Set([...words_1, ...words_2]);
      if (words.size === 0) {
        return false;
      }

      const dirty_words = await this.getDirtyWords();

      let results = dirty_words.filter((word) => words.has(word));

      return results.length > 0;
    },
    async checkSafetyAgreement() {
      if (!this.userOrderInformation) {
        await this.getUserOrderInformation();
      }
      const tier = this.userOrderInformation.tier;
      const permissions = await this.getFeaturePermissions();

      if (!permissions.features.PrivateImage.allowed_tiers.includes(tier)) {
        return true;
      }
      if (window.Cookies.get(SAFETY_AGREEMENT_KEY)) {
        return true;
      }

      if (await this.getSafetyAgreement()) {
        window.Cookies.set(SAFETY_AGREEMENT_KEY, true, { expires: 360 });
        return true;
      }

      if (!(await this.checkPromptDirtyWords(this.prompt))) {
        return true;
      }

      this.safetyPopup.isOpen = true;

      while (this.safetyPopup.isOpen) {
        await sleep(200);
      }

      try {
        if (this.safetyPopup.isAgreed) {
          await this.setSafetyAgreement();
          window.Cookies.set(SAFETY_AGREEMENT_KEY, true, { expires: 360 });
          return true;
        }

        return false;
      } finally {
        this.safetyPopup.isAgreed = false;
        this.safetyPopup.checkboxA = false;
        this.safetyPopup.checkboxB = false;
      }
    },
    async getSubscribeURL() {
      await this.getUserOrderInformation();
      const price_id = this.userOrderInformation.price_id;
      if (!price_id) {
        return null;
      }
      const params = new URLSearchParams({
        price_id: price_id,
        client_reference_id: Base64.encodeURI(
          JSON.stringify({ user_id: this.userOrderInformation.user_id }),
        ),
        allow_promotion_codes: true,
        current_url: window.location.href,
      });
      return `/pricing_table/checkout?${params.toString()}`;
    },
    openPopup(event, title, message, confirmText, url) {
      const popup = this.popup;

      popup.itemName = event;
      popup.title = title;
      popup.message = message;
      popup.confirmText = confirmText;
      popup.url = url;
      popup.isOpen = true;

      addPopupGtagEvent(popup.url, popup.itemName);
    },
    async upgradePopup(reason) {
      let event, title, confirmText, message, url;

      switch (reason) {
        case "NSFW_CONTENT":
          event = "refocus_nsfw_checker";
          title = "Upgrade Now";
          confirmText = "Upgrade";
          url = SUBSCRIPTION_URL;

          let hint;
          if (this.userOrderInformation.tier.toLowerCase() === "appsumo ltd tier 1") {
            hint = "lift these restrictions";
          } else {
            hint = "enable your private image storage";
          }

          message = `Potential NSFW content was detected in the generated image, \
            upgrade to ${NSFW_ALLOWED_TIERS_MESSAGE} to ${hint}. \
            Or join our ${AFFILIATE_PROGRAM} \
            to earn cash or credits and use it to upgrade to a higher plan.`;

          break;

        case "INSUFFICIENT_CREDITS":
          event = "refocus_insufficient_credits";
          title = "Upgrade Now";
          confirmText = "Upgrade";
          url = SUBSCRIPTION_URL;

          message = `You have ran out of your credits, please purchase more or upgrade to a \
            higher plan. Join our ${AFFILIATE_PROGRAM} to earn cash or credits.`;

          break;

        case "INSUFFICIENT_DAILY_CREDITS":
          event = "refocus_insufficient_daily_credits";
          title = "Subscribe Now";
          confirmText = "Subscribe Now";
          url = await this.getSubscribeURL();
          if (!url) {
            url = SUBSCRIPTION_URL;
          }

          message =
            "Your daily credits limit for the trial has been exhausted. \
            Subscribe now to unlock the daily restrictions.";

          break;

        case "REACH_CONCURRENCY_LIMIT":
          event = "refocus_reach_concurrency_limit";
          title = "Upgrade Now";
          confirmText = "Upgrade";
          url = SUBSCRIPTION_URL;

          const permissions = await this.getFeaturePermissions();

          await this.getUserOrderInformation();
          const tier = this.userOrderInformation.tier;

          const current_limit = permissions.limits[tier];
          if (!current_limit) {
            throw `user tier "${tier}" not found in the "limits" of permissions.`;
          }

          const max_concurrent_tasks = current_limit.max_concurrent_tasks;

          const getUnit = (limit) => (limit === 1 ? "task" : "tasks");

          message = `Your current plan allows only ${max_concurrent_tasks} concurrent \
            ${getUnit(max_concurrent_tasks)}.`;

          const higher_limits = permissions.upgradableLimits.filter(
            (item) => item.max_concurrent_tasks > max_concurrent_tasks,
          );
          if (higher_limits.length > 0) {
            message += " Upgrade to:";
            message += "<ul style='list-style: inside'>";
            for (let limit of higher_limits) {
              message += `<li><b>${limit.tier}</b> to run up to ${
                limit.max_concurrent_tasks
              } \
                ${getUnit(limit.max_concurrent_tasks)} simultaneously;</li>`;
            }
            message += "</ul>";
          }

          break;

        default:
          throw `Unknown upgrade reason: "${reason}".`;
      }

      this.openPopup(event, title, message, confirmText, url);
    },
    pushHistory(errorMessage = null) {
      this.historyResults.unshift({
        taskId: this.runningTaskId,
        params: this.generatingParams,
        paramsTable: this.turnObjParams(this.generatingParams),
        images: this.runningTaskResultImages,
        errorMessage: errorMessage,
      });
      if (this.historyResults.length > this.maxHistory) {
        this.historyResults.pop();
      }
    },
    mouseMoveToStylePreview(src) {
      this.styleOverlay.show = true;
      this.styleOverlay.src = src;
    },
    previewUserImage(image, imageSrcProperty, index = null) {
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (index != null) this[imageSrcProperty][index] = e.target.result;
          else this[imageSrcProperty] = e.target.result;
        };
        reader.readAsDataURL(file);
      }
    },
    clearPreview(imageSrcProperty, index = null) {
      if (index != null) this[imageSrcProperty][index] = null;
      else this[imageSrcProperty] = null;
    },
    editUserImage(event) {
      const targetNode = document.getElementById("inpaint-image-editor");
      const config = { childList: true, subtree: true };
      if (!this.toolObserver) {
        const callback = (mutationList, observer) => {
          const tools = document.querySelectorAll(".FIE_tools-item-wrapper");
          if (tools.length > 0) {
            tools.forEach((tool) => {
              const toolText =
                tool.querySelector("div > label > span").textContent;
              if (
                toolText === "Image" ||
                toolText === "Polygon" ||
                toolText === "Arrow"
              ) {
                tool.style.display = "none";
              }
            });
          }
        };
        this.toolObserver = new MutationObserver(callback);
        this.toolObserver.observe(targetNode, config);
      }
      if (!this.optionsObserver) {
        const optionsCallback = (mutationList, observer) => {
          const options = document.querySelectorAll(
            ".FIE_annotation-option-triggerer",
          );
          const penOptions = document.querySelector(".FIE_pen-tool-options");
          const lineOptions = document.querySelector(".FIE_line-tool-options");
          if (options.length > 0) {
            options.forEach((option) => {
              if (
                option.title === "Text alignment" ||
                option.title === "Text spacings" ||
                option.title === "Shadow" ||
                option.title === "Corner Radius"
              ) {
                option.style.display = "none";
              }
              if (!penOptions && !lineOptions) {
                if (option.title === "Stroke") {
                  option.style.display = "none";
                }
              }
            });
          }
        };
        this.optionsObserver = new MutationObserver(optionsCallback);
        this.optionsObserver.observe(targetNode, config);
      }
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          this.inpaintImageUploader = e.target.result;
          this.ImageEditor.render({
            source: e.target.result,
          });
        };
        reader.readAsDataURL(file);
      }
    },
    clearEdit() {
      this.inpaintImageUploader = null;
      if (this.ImageEditor) {
        this.ImageEditor.render();
      }
    },
    initializeFilerobot() {
      const container = document.getElementById("inpaint-image-editor");
      const config = {
        tabsIds: [
          FilerobotImageEditor.TABS.ANNOTATE,
          FilerobotImageEditor.TABS.RESIZE,
        ],
        defaultTabId: FilerobotImageEditor.TABS.ANNOTATE,
        defaultToolId: FilerobotImageEditor.TOOLS.PEN,
        observePluginContainerSize: true,
        annotationsCommon: {
          fill: "#FFFFFF",
          stroke: "#FFFFFF",
          opacity: 0.6,
        },
        Pen: {
          strokeWidth: 30,
        },
        Text: {
          text: "Change me!",
          fontSize: 28,
          fontStyle: "bold",
        },
        Line: {
          lineCap: "butt",
          strokeWidth: 16,
        },
        onBeforeSave: () => false,
        onSave: (imageInfo, designState) => {
          return false;
        },
      };
      this.ImageEditor = new window.FilerobotImageEditor(container, config);
    },
    toggleImageOptions() {
      if (this.showImageOptions) {
        this.showImageOptions = false;
      } else {
        this.panelName = "image_options";
        this.showImageOptions = true;
      }
    },
    turnObjParams(obj) {
      return Object.keys(obj)
        .filter((key) => {
          if (key === "uov_input_image") {
            return false;
          } else if (key === "ip_ctrls") {
            return false;
          } else if (key === "inpaint_input_image") {
            return false;
          } else {
            return obj[key];
          }
        })
        .map((key) => {
          if (key === "loras") {
            const lorasHtml = obj[key]
              .filter((lora) => lora.lora_model.toLowerCase() !== "none")
              .map((lora, index) => {
                return `${lora.lora_model}: ${lora.lora_weight}`;
              })
              .join("; ");
            return { name: key, value: lorasHtml };
          } else if (key === "style_selections") {
            return { name: key, value: obj[key].join(", ") };
          } else if (key === "advanced_options") {
            return {
              name: key,
              value: Object.keys(obj[key])
                .map((secondaryKey) => {
                  return `${secondaryKey}: ${obj[key][secondaryKey]}`;
                })
                .join("\n"),
            };
          } else if (obj[key]) {
            return { name: key, value: obj[key] };
          }
        });
    },
    packGenerationParameters() {
      let genParams = {
        task_id: this.runningTaskId,
        task_type: this.taskType,
        prompt: this.prompt,
        negative_prompt: this.negativePrompt,
        style_selections: this.selectedStyles,
        performance_selection: this.performance,
        aspect_ratios_selection: this.aspectRatio,
        image_number: this.imageNumber,
        image_seed: this.isRandom ? -1 : this.randomSeed,
        sharpness: this.taskType === "sdxl"? this.imageSharpness: this.clipSkip,
        guidance_scale: this.guidanceScale,
        base_model: this.baseModel,
        refiner_model: this.refiner,
        loras: this.selectedLoraModels.map((model, index) => ({
          lora_model: model,
          lora_weight: this.selectedLoraWeights[index],
        })),
        advanced_options: {
          sampler_name: this.selectedSampler,
          scheduler_name: this.selectedScheduler,
        },
      };
      if (this.refiner !== "None") {
        genParams.refiner_switch = this.refinerSwitch;
      }
      if (this.forcedSteps >= 1) {
        genParams.advanced_options.overwrite_step = this.forcedSteps;
      }
      if (this.showImageOptions) {
        if (this.imageOptionTab === "uov") {
          if (this.uovImageFile) {
            genParams.input_image_checkbox = this.showImageOptions;
            genParams.current_tab = this.imageOptionTab;
            genParams.uov_method = this.uovSelection;
            genParams.uov_input_image = {
              encoded_image: this.uovImageFile,
            };
          }
        } else if (this.imageOptionTab === "ip") {
          for (const [
            index,
            imagePromptUploader,
          ] of this.imagePromptImages.entries()) {
            if (imagePromptUploader) {
              genParams.input_image_checkbox = this.showImageOptions;
              genParams.current_tab = this.imageOptionTab;
              if (typeof genParams.ip_ctrls === "undefined") {
                genParams.ip_ctrls = [];
              }
              ip_control = {
                ip_image: {
                  encoded_image: imagePromptUploader,
                },
                ip_stop: this.imagePrompts[index].params.stopAt,
                ip_weight: this.imagePrompts[index].params.weight,
                ip_type: this.imagePrompts[index].params.controlType,
              };
              genParams.ip_ctrls.push(ip_control);
            }
          }
        } else if (this.imageOptionTab === "inpaint") {
          if (this.inpaintImageUploader) {
            genParams.input_image_checkbox = this.showImageOptions;
            genParams.current_tab = this.imageOptionTab;
            const imageData = this.ImageEditor.getCurrentImgData();
            const inpaintMask = drawAnnotationsOnCanvas(
              imageData.designState,
              imageData.imageData.width,
              imageData.imageData.height,
            ).toDataURL("image/png");
            genParams.inpaint_input_image = {
              image: {
                encoded_image: this.inpaintImageUploader,
              },
              mask: {
                encoded_image: inpaintMask,
              },
            };
            if (this.inpaintSelection === "Inpaint or Outpaint (default)") {
              genParams.outpaint_selections = this.outpaintDirection;
            } else if (
              this.inpaintSelection ===
              "Improve Detail (face, hand, eyes, etc.)"
            ) {
              genParams.inpaint_additional_prompt =
                this.inpaintImprovementPrompt;
            } else if (
              this.inpaintSelection ===
              "Modify Content (add objects, change background, etc.)"
            ) {
              genParams.inpaint_additional_prompt =
                this.inpaintModificationPrompt;
            }
          }
        }
      }
      return genParams;
    },
    async startToGenerate(retry) {
      if (this.taskType === "sd3") {
          const is_allowed = await this.checkPermission("SD3");
          if (!is_allowed) {
            return;
          }
      }

      if (this.taskType === "flux") {
          const is_allowed = await this.checkPermission("Flux");
          if (!is_allowed) {
            return;
          }
      }
      if (!(await this.checkSafetyAgreement())) {
        return;
      }
      if (retry > 5) {
        this.generating = false;
        return;
      }
      let socket = null;
      let wsProtocol = "wss";
      if (location.protocol == "http:") {
        wsProtocol = "ws";
      }
      if (this.generating && this.runningTaskId) {
        socket = new WebSocket(
          `${wsProtocol}://${window.location.host}/api/focus/ws/generate?` +
            new URLSearchParams({
              task_id: this.runningTaskId,
            }),
        );
      } else {
        this.runningTaskId = randomId();
        socket = new WebSocket(
          `${wsProtocol}://${window.location.host}/api/focus/ws/generate?` +
            new URLSearchParams({
              new_task_id: this.runningTaskId,
              task_type: this.taskType,
            }),
        );
        reportSpendCreditsEvent(
          "refocus_generate_button",
          this.estimateConsume.inference,
        );
      }
      this.generating = true;
      this.runningTaskMessage = "Preparing...";
      socket.onopen = (event) => {
        this.generatingParams = this.packGenerationParameters();
        socket.send(JSON.stringify(this.generatingParams));
        this.runningTaskPreviewImage = "";
        this.runningTaskResultImages = [];
        this.stopLoading = false;
        this.skipLoading = false;
      };
      socket.onmessage = async (event) => {
        const status = JSON.parse(event.data);
        this.runningTaskStatus = status.status;
        this.runningTaskProgress = status.progress;
        this.runningTaskMessage = status.message;
        this.runningTaskReturnUrl = status.is_url;
        this.runningTaskQueueLength = status.queue_length;
        this.runningTaskQueuePosiiton = status.queue_position;
        if (status.status === "queueing" || status.status === "preparing") {
          this.waiting = true;
        } else {
          this.waiting = false;
        }
        if (status.status === "preview" && status.images.length > 0) {
          if (status.is_url) {
            this.runningTaskPreviewImage = status.images[0].image_url;
          } else {
            this.runningTaskPreviewImage = status.images[0].encoded_image;
          }
        }
        if (status.status === "results" && status.images.length > 0) {
          if (status.is_url) {
            this.runningTaskPreviewImage =
              status.images[status.images.length - 1].image_url;
            this.runningTaskResultImages = status.images.map((item) => {
              return { src: item.image_url, id: item.image_id };
            });
          } else {
            this.runningTaskPreviewImage =
              status.images[status.images.length - 1].encoded_image;
            this.runningTaskResultImages = status.images.map((item) => {
              return { src: item.encoded_image, id: item.image_id };
            });
          }
          await this.checkNSFW(status.is_nsfw);
        }
        if (status.status === "skipped") {
          this.skipLoading = false;
        }
        if (status.status === "finish") {
          this.stopLoading = false;
          this.generating = false;
          if (status.is_url) {
            this.runningTaskResultImages = status.images.map((item) => {
              return { src: item.image_url, id: item.image_id };
            });
          } else {
            this.runningTaskResultImages = status.images.map((item) => {
              return { src: item.encoded_image, id: item.image_id };
            });
          }
          await this.checkNSFW(status.is_nsfw);
          this.pushHistory();
        }
        if (status.status === "failed") {
          this.stopLoading = false;
          this.generating = false;
          this.pushHistory(status.message);
        }
        if (status.status.startsWith("MonitorException.")) {
          this.stopLoading = false;
          this.generating = false;

          const reason = status.status.split(".")[1];
          await this.upgradePopup(reason);
        }
      };
      socket.onerror = (error) => {
        // TODO: show error message
        console.log("WebSocket error:", error);
      };
      socket.onclose = (event) => {
        console.log("WebSocket connection closed:", event.code);
        if (this.generating) {
          if (this.runningTaskId) {
            setTimeout(() => {
              this.startToGenerate(retry + 1);
            }, 3000);
          } else {
            this.generating = false;
            this.stopLoading = false;
          }
        }
      };
    },
    skipGeneration() {
      this.skipLoading = true;
      fetch(
        "/api/focus/skip?" +
          new URLSearchParams({
            task_id: this.runningTaskId,
          }),
        {
          method: "POST",
          headers: {
            Accept: "application/json",
            "Content-Type": "application/json",
          },
        },
      )
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then((result) => {
          if (result.status === "failed") {
            this.skipLoading = false;
            // TODO: show error message
          }
        })
        .catch((error) => {
          this.skipLoading = false;
          console.error("Post skip error:", error);
        });
    },
    stopGeneration() {
      this.stopLoading = true;
      fetch(
        "/api/focus/stop?" +
          new URLSearchParams({
            task_id: this.runningTaskId,
          }),
        {
          method: "POST",
          headers: {
            Accept: "application/json",
          },
        },
      )
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then((result) => {
          if (result.status === "failed") {
            this.stopLoading = false;
            // TODO: show error message
          }
        })
        .catch((error) => {
          this.stopLoading = false;
          console.error("Post skip error:", error);
        });
    },
    findClosestItem(arr, attributeName, benchmark) {
      if (!arr.length) {
        throw new Error("Array is empty");
      }

      let closest = arr[0];
      let smallestDiff = Math.abs(arr[0][attributeName] - benchmark);

      for (let i = 1; i < arr.length; i++) {
        const diff = Math.abs(arr[i][attributeName] - benchmark);
        if (diff < smallestDiff) {
          smallestDiff = diff;
          closest = arr[i];
        }
      }
      return closest;
    },
    describeImageLocal() {
      this.describeImageLoading = true;
      fetch("/api/focus/describe/local", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          mode: this.contentTypeSelection,
          image: {
            encoded_image: this.describeImageUploader,
          },
        }),
      })
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then((result) => {
          this.prompt = result.prompt;
          this.selectedStyles = result.styles;
          this.describeImageLoading = false;
          if (!result.image_id) {
            // TODO: show error message
          }
        })
        .catch((error) => {
          this.describeImageLoading = false;
          console.error("Post describe error:", error);
        });
    },
    getGptVisionTaskStatus(task_id, retries = 5, backoff = 300) {
      if (retries <= 0) {
        this.describeImageLoading = false;
        notifier.alert(
          "Could not get GPT vision task progress. Please try again later.",
        );
        console.warn("Get gpt vision task progress retries exhausted");
        return;
      }
      fetch(
        "/api/v3/gpt/vision/prompt?" +
          new URLSearchParams({
            task_id: task_id,
          }),
        {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
        },
      )
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then(async (result) => {
          if (result.status === "failed") {
            if (result.message.startsWith("MonitorException.")) {
              const reason = result.message.split(".")[1];
              await this.upgradePopup(reason);
            } else {
              notifier.warning("Failed to generate prompt uisng GPT vision.");
            }
            this.describeImageLoading = false;
            return;
          } else if (result.status === "finished") {
            this.describeImageLoading = false;
            if (result.nsfw) {
              notifier.warning(
                "Image violates GPT vision policy. We could not proceed with the request nor refund your credits.",
              );
            } else {
              this.prompt = result.prompt;
              this.gptVisionTask.result = result.prompt;
              let [width, height] = await this.getImageResolution(
                this.describeImageUploader,
              );
              const ratio = width / height;
              const ratioItem = this.findClosestItem(
                this.aspectRatiosNumber,
                "ratio",
                ratio,
              );
              this.aspectRatio = ratioItem.text;
            }
          } else if (result.status === "started") {
            setTimeout(() => {
              this.getGptVisionTaskStatus(task_id);
            }, 1000);
          } else if (result.status === "queued") {
            this.gptVisionTask.queueLength = result.queue_status.queueLength;
            this.gptVisionTask.queuePosition = result.queue_status.position;
            setTimeout(() => {
              this.getGptVisionTaskStatus(task_id);
            }, 1000);
          } else if (result.status === "unknown") {
            setTimeout(() => {
              this.getGptVisionTaskStatus(task_id, retries - 1);
            }, 1000);
          } else {
            this.describeImageLoading = false;
            notifier.alert(
              "Could not get GPT vision task progress. Please try again later.",
            );
          }
          if (result.status !== "queued") {
            this.gptVisionTask.queueLength = 0;
            this.gptVisionTask.queuePosition = 0;
          }
        })
        .catch((error) => {
          if (
            retries > 0 &&
            (error.name === "AbortError" || error.name === "NetworkError")
          ) {
            console.warn(`Retrying (${retries} more attempts)`, error);
            setTimeout(
              () =>
                this.getGptVisionTaskStatus(task_id, retries - 1, backoff * 2),
              backoff,
            );
          } else {
            this.describeImageLoading = false;
            console.error("Get gpt vision task progress error:", error);
          }
        });
    },
    describeImageGptVision() {
      this.describeImageLoading = true;
      this.describeErrorMessage = null;
      fetch("/api/v3/gpt/vision/prompt", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: {
            encoded_image: this.describeImageUploader,
          },
        }),
      })
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          } else if (response.status === 451) {
            response.json().then((result) => {
              this.describeErrorMessage = result.detail.message;
            });
          }
          return Promise.reject(response);
        })
        .then((result) => {
          this.gptVisionTask.task_id = result.data.task_id;
          setTimeout(() => {
            this.getGptVisionTaskStatus(result.data.task_id);
          }, 1000);
        })
        .catch((error) => {
          this.describeImageLoading = false;
          console.error("Post describe error:", error);
        });
    },
    toggleRandom() {
      if (this.isRandom) {
        this.randomSeed = "";
      }
    },
    expandImage(index, images) {
      this.previewIndex = index;
      this.previewImages = images;
      this.previewDialog = true;
    },
    likeOrUnlike(imageId) {
      let like = true;
      if (this.likeButtonColor[imageId]) {
        like = false;
      }
      fetch("/api/focus/like", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_id: imageId,
          like: like,
        }),
      })
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then((result) => {
          if (result.success) {
            if (result.liked) {
              this.likeButtonColor[imageId] = "red-lighten-1";
            } else {
              this.likeButtonColor[imageId] = "";
            }
          }
        })
        .catch((error) => {
          console.error("Post like error:", error);
        });
    },
    favoriteOrUnfavorite(imageId) {
      let favorite = true;
      if (this.favoriteButtonColor[imageId]) {
        favorite = false;
      }
      fetch("/api/focus/favorite", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_id: imageId,
          favorite: favorite,
        }),
      })
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then((result) => {
          if (result.success) {
            if (result.favorited) {
              this.favoriteButtonColor[imageId] = "orange-lighten-1";
            } else {
              this.favoriteButtonColor[imageId] = "";
            }
          }
        })
        .catch((error) => {
          console.error("Post favorite error:", error);
        });
    },
    shareOrUnshare(imageId) {
      let share = true;
      if (this.shareButtonColor[imageId]) {
        share = false;
      }
      fetch("/api/focus/share", {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image_id: imageId,
          share: share,
        }),
      })
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then((result) => {
          if (result.success) {
            if (result.shared) {
              this.shareButtonColor[imageId] = "blue-lighten-1";
            } else {
              this.shareButtonColor[imageId] = "";
            }
          }
        })
        .catch((error) => {
          console.error("Post share error:", error);
        });
    },
    updateImagePromptDefault(index, styleType) {
      this.imagePrompts[index].params.stopAt =
        this.imagePromptDefaultValues[styleType].stop;
      this.imagePrompts[index].params.weight =
        this.imagePromptDefaultValues[styleType].weight;
    },
    updateDefaultOptions(preset, taskType = null, callback = null) {
      this.loadingPreset = true;
      if (this.selectedPreset === preset) {
        this.selectingPreset = false;
        this.loadingPreset = false;
        return;
      }
      taskType = taskType || this.taskType;
      fetch(
        "/api/focus/default_options?" +
          new URLSearchParams({
            preset: preset,
            task_type: taskType,
          }),
        {
          method: "GET",
        },
      )
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then((options) => {
          this.defaultOptions = options;
          this.aspectRatios = options.aspect_ratios.options;
          this.aspectRatio = options.aspect_ratios.default;
          this.aspectRatiosNumber = options.aspect_ratios.options.map(
            (item) => {
              const regex = /(\d+)x(\d+) \((\d+):(\d+)\)/;
              const match = item.match(regex);

              if (match) {
                const width = parseInt(match[1], 10);
                const height = parseInt(match[2], 10);
                const ratio = match[3] / match[4];

                return { width, height, ratio, text: item };
              } else {
                throw new Error(
                  "Input string does not match the expected format",
                );
              }
            },
          );
          this.styles = options.styles.options.map((style) => {
            return {
              name: style.style_name,
              image: style.style_preview,
            };
          });
          this.selectedStyles = options.styles.default_list;
          this.baseModels = options.base_models.options;
          this.baseModel = options.base_models.default;
          this.refiners = options.refiner_models.options;
          this.refiner = options.refiner_models.default;
          this.refinerSwitch = options.refiner_switch;
          this.numLoras = options.loras.length;
          this.uovMethods = options.uovs.options;
          this.uovSelection = options.uovs.default;
          this.ipTypes = options.ip_types.options;
          this.numImagePrompts = options.num_image_prompts;
          this.contentTypeSelection = options.content_types.default;
          this.contentTypes = options.content_types.options;
          this.imagePromptDefaultValues = options.ip_default_options;
          this.selectedLoraModels = [];
          this.selectedLoraWeights = [];
          for (let i = 0; i < this.numLoras; i++) {
            this.selectedLoraModels.push(options.loras[i].lora_name.default);
            this.selectedLoraWeights.push(options.loras[i].lora_weight);
            this.loraModels = options.loras[i].lora_name.options;
          }
          for (let i = 0; i < this.numImagePrompts; i++) {
            this.imagePrompts.push({
              params: {
                stopAt:
                  options.ip_default_options[options.ip_types.default].stop,
                weight:
                  options.ip_default_options[options.ip_types.default].weight,
                controlType: options.ip_types.default,
              },
            });
            this.imagePromptImages.push(null);
          }
          this.selectedPreset = preset;
          this.availablePresets = options.presets.options;
          for (const presetName of this.availablePresets) {
            if (!(presetName in this.presetName)) {
              this.presetName[presetName] = presetName;
            }
            if (!(presetName in this.presetColor)) {
              this.presetColor[presetName] = "grey-darken-1";
            }
          }
          this.guidanceScale = options.cfg_scale;
          if (taskType === "sdxl") {
            this.imageSharpness = options.sample_sharpness;
          } else {
            this.clipSkip = options.sample_sharpness;
          }
          this.selectedSampler = options.sampler.default;
          this.availableSamplers = options.sampler.options;
          this.selectedScheduler = options.scheduler.default;
          this.availableSchedulers = options.scheduler.options;
          //if (options.prompt) {
          //  this.prompt = options.prompt;
          //}
          this.negativePrompt = options.negative_prompt;

          this.nonLCMArguments = {};
          this.performanceOptions = options.performances.options;
          this.performance = options.performances.default;
          this.forcedSteps = options.steps;

          this.selectingPreset = false;
          this.loadingPreset = false;

          this.sd3.baseModel = options.sd3.base_models.default;
          this.sd3.baseModels = options.sd3.base_models.options;
          this.sd3.aspectRatio = options.sd3.aspect_ratios.default;
          this.sd3.aspectRatios = options.sd3.aspect_ratios.options;

          if (typeof callback === "function") {
            callback(options);
          }
        })
        .catch((error) => {
          console.error("Get default value error:", error);
          this.selectingPreset = false;
          this.loadingPreset = false;
          if (!this.selectedPreset) {
            this.selectedPreset = "default";
          }
        });
    },
    getConsumeText(consume, image_number = null) {
      const inference = consume.inference;
      const discount = consume.discount;

      const real_inference =
        discount === 0 ? inference : Math.ceil(inference * (1 - discount));
      const credit_unit = `credit${real_inference === 1 ? "" : "s"}`;

      let result =
        discount === 0
          ? `Estimated ${real_inference} ${credit_unit}`
          : `Estimated <del>${inference}</del> ${real_inference} ${credit_unit}`;

      if (image_number) {
        const image_unit = `image${image_number === 1 ? "" : "s"}`;
        result = `${result} for ${image_number} ${image_unit}`;
      }
      return `(${result})`;
    },
    getShapeCeil(width, height) {
      return Math.ceil(Math.sqrt(width * height) / 64.0) * 64.0;
    },
    getTargetResolution() {
      const [width, height] = this.aspectRatio
        .replace("x", " ")
        .split(" ")
        .slice(0, 2);
      return [Number(width), Number(height)];
    },
    getImageResolution(src) {
      return new Promise((resolve, _) => {
        const img = new Image();
        img.src = src;
        img.onload = () => {
          resolve([img.naturalWidth, img.naturalHeight]);
        };
      });
    },
    getStepsCoefficient() {
      const mapping = {
        Speed: 2,
        Quality: 4,
        "Extreme Speed": 1,
        Turbo: 1,
      };
      return mapping[this.performance];
    },
    async requestCreditsConsumption(args) {
      const body = {
        ...args,
        link_params: {},
        mutipliers: {},
        link_mutipliers: {},
      };

      const response = await fetch("/api/tasks/credits_consumption", {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ functions: args }),
      });
      if (response.status != 200) {
        return { inference: "-", discount: 0 };
      }
      const result = await response.json();
      return {
        inference: result.inference,
        discount: result.discount ? result.discount : 0,
      };
    },
    async getUserOrderInformation() {
      return fetch("/api/order_info", {
        method: "GET",
        credentials: "include",
      })
        .then((response) => {
          if (response.status === 200) {
            return response.json();
          }
          return Promise.reject(response);
        })
        .then((data) => {
          this.userOrderInformation = data;
        })
        .catch((error) => {
          console.error("getUserOrderInformation error:", error);
        });
    },
    async getFeaturePermissions() {
      if (!this._featurePermissions) {
        const response = await fetch("/config/feature_permissions");
        const body = await response.json();

        this._featurePermissions = {
          generate: body.generate,
          buttons: Object.fromEntries(
            body.buttons.map((item) => [item.name, item]),
          ),
          features: Object.fromEntries(
            body.features.map((item) => [item.name, item]),
          ),
          limits: Object.fromEntries(
            body.limits.map((item) => [item.tier, item]),
          ),
        };

        this._featurePermissions.upgradableLimits = UPGRADABLE_TIERS.map(
          (item) => this._featurePermissions.limits[item],
        );
      }
      return this._featurePermissions;
    },
    async getImageOptionsConsumeArgs() {
      if (!this.showImageOptions) {
        return;
      }
      if (this.imageOptionTab === "uov") {
        if (!this.uovImageFile) {
          return;
        }
        let [width, height] = await this.getImageResolution(this.uovImageFile);
        if (this.uovSelection.includes("Vary")) {
          const shapeCeil = this.getShapeCeil(width, height);
          if (shapeCeil <= 1024) {
            width = 1024;
            height = 1024;
          } else if (shapeCeil >= 2048) {
            width = 2048;
            height = 2048;
          }
          return {
            "fooocus.vary": {
              params: {
                width: width,
                height: height,
                steps_coefficient: this.getStepsCoefficient(),
                image_number: this.imageNumber,
              },
            },
          };
        }
        if (this.uovSelection.includes("Upscale")) {
          const scale = this.uovSelection.includes("1.5") ? 1.5 : 2.0;
          width *= scale;
          height *= scale;

          let is_fast = this.uovSelection.includes("Fast");

          let shapeCeil = this.getShapeCeil(width, height);
          if (shapeCeil <= 1024) {
            width = 1024;
            height = 1024;
          } else if (shapeCeil > 2800) {
            is_fast = true;
          }
          let steps_coefficient = 0;
          let image_number = 1;
          if (!is_fast) {
            steps_coefficient = this.getStepsCoefficient();
            image_number = this.imageNumber;
          }

          return {
            "fooocus.upscale": {
              params: {
                width: width,
                height: height,
                steps_coefficient: steps_coefficient,
                image_number: image_number,
              },
            },
          };
        }
        return;
      }
      if (this.imageOptionTab === "inpaint") {
        if (!this.inpaintImageUploader) {
          return;
        }
        let [width, height] = await this.getImageResolution(
          this.inpaintImageUploader,
        );

        if (this.inpaintSelection === "Inpaint or Outpaint (default)") {
          let scale = 0;
          if (this.outpaintDirection.includes("Top")) {
            scale += 0.3;
          }
          if (this.outpaintDirection.includes("Bottom")) {
            scale += 0.3;
          }
          height += height * scale;

          scale = 0;
          if (this.outpaintDirection.includes("Left")) {
            scale += 0.3;
          }
          if (this.outpaintDirection.includes("Right")) {
            scale += 0.3;
          }
          width += height * scale;
        }
        return {
          "fooocus.inpaint": {
            params: {
              width: Math.floor(width),
              height: Math.floor(height),
              steps_coefficient: this.getStepsCoefficient(),
              image_number: this.imageNumber,
            },
          },
        };
      }
      if (this.imageOptionTab === "ip") {
        let ip_ctrls = 0;
        for (const imagePromptUploader of this.imagePromptImages) {
          if (imagePromptUploader) {
            ip_ctrls += 1;
          }
        }
        const [width, height] = this.getTargetResolution();
        return {
          fooocus: {
            params: {
              width: width,
              height: height,
              steps_coefficient: this.getStepsCoefficient(),
              ip_ctrls: ip_ctrls,
              image_number: this.imageNumber,
            },
          },
        };
      }
      return;
    },
    async updateEstimateConsume(_) {
      let args = await this.getImageOptionsConsumeArgs();
      if (!args) {
        const [width, height] = this.getTargetResolution();
        if (this.taskType === "sd3") {
          args = {
            "fooocus.sd3": {
              params: {
                width: width,
                height: height,
                steps_coefficient: this.getStepsCoefficient(),
                ip_ctrls: 0,
                image_number: this.imageNumber,
                ratio: 1.5,
              },
            },
          };
        } else if (this.taskType === "flux") {
          args = {
            "fooocus.flux": {
              params: {
                width: width,
                height: height,
                steps_coefficient: this.getStepsCoefficient(),
                ip_ctrls: 0,
                image_number: this.imageNumber,
                ratio: 2,
              },
            },
          };
        } else {
          args = {
            fooocus: {
              params: {
                width: width,
                height: height,
                steps_coefficient: this.getStepsCoefficient(),
                ip_ctrls: 0,
                image_number: this.imageNumber,
                ratio: 1,
              },
            },
          };
        }
      }
      if (JSON.stringify(args) === JSON.stringify(this.estimateConsume.args)) {
        return;
      }
      this.estimateConsume.args = args;
      if (this.estimateConsume.timeoutId !== null) {
        clearTimeout(this.estimateConsume.timeoutId);
      }
      this.estimateConsume.timeoutId = setTimeout(async () => {
        let request_args = this.estimateConsume.args;
        this.estimateConsume.imageNumber =
          Object.values(request_args)[0].params.image_number;
        const result = await this.requestCreditsConsumption(request_args);
        this.estimateConsume.inference = result.inference;
        this.estimateConsume.discount = result.discount;
        this.estimateConsume.timeoutId = null;
      }, 1000);
    },
    async updateBlipEstimateConsume(_) {
      if (!this.showImageOptions) {
        this.estimateBlipConsume = { inference: "-", discount: 0 };
        return;
      }
      if (this.imageOptionTab != "desc") {
        this.estimateBlipConsume = { inference: "-", discount: 0 };
        return;
      }
      if (!this.describeImageUploader) {
        this.estimateBlipConsume = { inference: "-", discount: 0 };
        return;
      }
      let [width, height] = await this.getImageResolution(
        this.describeImageUploader,
      );
      args = {
        "fooocus.describe.blip": {
          params: {
            width: width,
            height: height,
          },
        },
      };
      const result = await this.requestCreditsConsumption(request_args);
      this.estimateBlipConsume.inference = result.inference;
      this.estimateBlipConsume.discount = result.discount;
    },
    //updateSD3EstimateConsume() {
    //  const estimateConsume = { discount: 0.25, imageNumber: this.imageNumber };
    //  if (this.sd3.baseModel === "sd3") {
    //    estimateConsume.inference = 26 * this.imageNumber;
    //  } else if (this.sd3.baseModel === "sd3-turbo") {
    //    estimateConsume.inference = 16 * this.imageNumber;
    //  }
    //  this.sd3.estimateConsume = estimateConsume;
    //},
    async requestGptVisionCreditsConsumption(args) {
      const response = await fetch("/api/v3/gpt/vision/prompt/credits", {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(args),
      });
      if (response.status != 200) {
        return { inference: "-", discount: 0 };
      }
      const result = await response.json();
      return {
        inference: result.inference,
        discount: result.discount ? result.discount : 0,
      };
    },
    async updateGptVisionEstimateConsume(_) {
      if (!this.showImageOptions) {
        this.estimateGptVisionConsume = { inference: "-", discount: 0 };
        return;
      }
      if (this.imageOptionTab != "desc") {
        this.estimateGptVisionConsume = { inference: "-", discount: 0 };
        return;
      }
      if (!this.describeImageUploader) {
        this.estimateGptVisionConsume = { inference: "-", discount: 0 };
        return;
      }
      let [width, height] = await this.getImageResolution(
        this.describeImageUploader,
      );
      const args = {
        width: width,
        height: height,
      };
      const result = await this.requestGptVisionCreditsConsumption(args);
      this.estimateGptVisionConsume.inference = result.inference;
      this.estimateGptVisionConsume.discount = result.discount;
    },
    openSubscriptionPage() {
      addUpgradeGtagEvent(this.popup.url, this.popup.itemName);
      window.open(this.popup.url, "_blank");
      this.popup.isOpen = false;
    },
    updateLCMOptions() {
      if (this.isLCMMode) {
        this.nonLCMArguments = {
          refiner: this.refiner,
          selectedSampler: this.selectedSampler,
          selectedScheduler: this.selectedScheduler,
          imageSharpness: this.imageSharpness,
          guidanceScale: this.guidanceScale,
        };
        this.refiner = "None";
        this.selectedSampler = "lcm";
        this.selectedScheduler = "lcm";
        this.imageSharpness = 0;
        this.guidanceScale = 1;
      } else if (Object.keys(this.nonLCMArguments).length != 0) {
        this.refiner = this.nonLCMArguments.refiner;
        this.selectedSampler = this.nonLCMArguments.selectedSampler;
        this.selectedScheduler = this.nonLCMArguments.selectedScheduler;
        this.imageSharpness = this.nonLCMArguments.imageSharpness;
        this.guidanceScale = this.nonLCMArguments.guidanceScale;
        this.nonLCMArguments = {};
      }
    },
    async postSD3txt2img() {
      const body = {
        task_id: this.runningTaskId,
        prompt: this.prompt,
        aspect_ratio: this.sd3.aspectRatio,
        negative_prompt: this.negativePrompt,
        model: this.sd3.baseModel,
        seed: this.isRandom ? 0 : this.randomSeed,
        image_number: this.imageNumber,
      };

      return fetch("/api/v3/internal/stability/sd3/txt2img", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    },
    async postSD3img2img() {
      const body = {
        task_id: this.runningTaskId,
        prompt: this.prompt,
        negative_prompt: this.negativePrompt,
        model: this.sd3.baseModel,
        seed: this.isRandom ? 0 : this.randomSeed,
        image_number: this.imageNumber,
        image: this.imagePromptImages[0],
        strength: this.sd3.strength,
      };

      return fetch("/api/v3/internal/stability/sd3/img2img", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    },
    async _openTierCheckPopup(reason, name, full_name, allowed_tiers) {
      if (reason === "TIER_NOT_ALLOWED") {
        const allowed_tiers_message = _joinTiers(allowed_tiers);
        const message = `<b>${full_name}</b> is not available in the current plan. \
                        Please upgrade to ${allowed_tiers_message} to use it.`;

        this.openPopup(
          `refocus_${name.toLowerCase()}_tier_checker`,
          "Upgrade Now",
          message,
          "Upgrade",
          SUBSCRIPTION_URL,
        );
        return;
      }
      if (reason === "TRIAL_NOT_ALLOWED") {
        const message =
          `<b>${full_name}</b> is not available in trial. Subscribe now to use it.`;
        const url = await this.getSubscribeURL();
        if (!url) {
          url = SUBSCRIPTION_URL;
        }
        this.openPopup(
          `refocus_${name.toLowerCase()}_trialing_checker`,
          "Subscribe Now",
          message,
          "Subscribe Now",
          url,
        );
      }
    },
    async _checkPermission(allowed_tiers, use_cache = true, trial_allowed = true) {
      if (!use_cache) {
        await this.getUserOrderInformation();
      }
      const order_info = this.userOrderInformation;
      if (!allowed_tiers.includes(order_info.tier)) {
        return { allowed: false, reason: "TIER_NOT_ALLOWED" };
      }
      if (!trial_allowed && order_info.trialing) {
        return { allowed: false, reason: "TRIAL_NOT_ALLOWED" };
      }
      return { allowed: true };
    },
    async checkPermission(name, full_name = null, trial_allowed = true) {
      const permissions = await this.getFeaturePermissions();
      const allowed_tiers = permissions.buttons[name].allowed_tiers;

      let result = await this._checkPermission(allowed_tiers, true, trial_allowed);
      if (!result.allowed) {
        result = await this._checkPermission(allowed_tiers, false, trial_allowed);
      }
      if (result.allowed) {
        return true;
      }
      this._openTierCheckPopup(result.reason, name, full_name ? full_name : name, allowed_tiers)
      return false;
    },
    async generateSD3() {
      const is_allowed = await this.checkPermission("SD3", "Stable Diffusion 3", false);
      if (!is_allowed) {
        return;
      }

      let errorMessage = null;
      let response;

      this.generating = true;
      this.runningTaskMessage = "Generating...";
      this.runningTaskResultImages = [];
      this.runningTaskId = randomId();

      try {
        if (this.imagePromptImages[0]) {
          response = await this.postSD3img2img();
        } else {
          response = await this.postSD3txt2img();
        }
        if (!response.ok) {
          const status_code = response.status;
          if (status_code === 402 || status_code === 429) {
            const content = await response.json();
            if (content.detail.need_upgrade) {
              await this.upgradePopup(content.detail.reason);
            }
          } else if (status_code === 451) {
            const content = await response.json();
            errorMessage = content.detail.message;
          } else if (status_code === 424) {
            const content = await response.json();
            const detail = content.detail;
            errorMessage = `Error "${
              detail.name
            }" from Stability AI: ${detail.errors.join(" ")}`;
          } else {
            errorMessage = await response.text();
          }
          return;
        }

        content = await response.json();
        this.runningTaskResultImages = content.images.map((item) => ({
          src: item.image,
          id: 0,
        }));
      } finally {
        this.generating = false;
        this.pushHistory(errorMessage);
        this.runningTaskId = "";
      }
    },
  },
  computed: {
    isLCMMode() {
      return this.performance === "Extreme Speed";
    },
    performance: {
      get() {
        return this._performance;
      },
      set(value) {
        this._performance = value;
        this.updateLCMOptions();
      },
    },
    numImagePrompts: {
      get() {
        return this._numImagePrompts;
      },
      set(value) {
        this._numImagePrompts = value;
      },
    },
  },
  watch: {
    imageOptionTab(newTab, oldTab) {
      if (oldTab === "inpaint" && newTab !== "inpaint") {
        this.clearEdit();
        this.inpaintImageFiles = [];
      }
    },
    async userOrderInformation(newInfo, oldInfo) {
      if (newInfo) {
        await reportIdentity(newInfo.user_id, newInfo.email);
      }
    },
    taskType(newType, oldType) {
      if (newType.toLowerCase() !== "sdxl" && this.imageOptionTab !== "desc") {
        this.imageOptionTab = "desc";
      }
    },
  },
  created() {
    for (let name of [
      "showImageOptions",
      "imageOptionTab",
      "uovImageFile",
      "uovSelection",
      "performance",
      "inpaintImageUploader",
      "inpaintSelection",
      "outpaintDirection",
      "aspectRatio",
      "imageNumber",
      "taskType",
    ]) {
      this.$watch(name, this.updateEstimateConsume);
    }
    this.$watch("imagePromptImages", this.updateEstimateConsume, {
      deep: true,
    });

    for (let name of [
      "showImageOptions",
      "imageOptionTab",
      "describeImageUploader",
    ]) {
      this.$watch(name, this.updateGptVisionEstimateConsume);
    }

    //for (let name of ["sd3.baseModel", "imageNumber"]) {
    //  this.$watch(name, this.updateSD3EstimateConsume);
    //}
  },
  async mounted() {
    this.updateDefaultOptions("default", this.taskType, () => {
      this.initializeFilerobot();
      this.startIntroJS();
    });
    this.getUserOrderInformation();
  },
})
  .use(vuetify)
  .mount("#app");
