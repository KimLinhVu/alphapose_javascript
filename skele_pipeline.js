import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import * as onnx from "onnxruntime-node";
import ffmpeg from "fluent-ffmpeg";
import path from "path";
import { heatmapToCoordSimpleRegress } from "./transformers.js";

const batchSize = 1;

// Extract frames from a video and save them as PNG files
async function extractFrames(videoPath) {
	return new Promise((resolve, reject) => {
		// Directory to store extracted frames
		const frameDir = path.join("frames");
		if (!fs.existsSync(frameDir)) {
			fs.mkdirSync(frameDir);
		}

		const outputPath = path.join(frameDir, "frame%04d.png");

		ffmpeg(videoPath)
			.output(outputPath)
			.on("end", () => {
				const files = fs.readdirSync(frameDir);
				resolve(
					files.length > 0 ? files.map((file) => path.join(frameDir, file)) : []
				);
			})
			.on("error", (err) => reject(`Error extracting frames: ${err.message}`))
			.run();
	});
}

// Process video frames into tensors for model input
async function processVideo(inputPath) {
	try {
		const frames = await extractFrames(inputPath);

		// Convert each frame into a tensor and resize
		const frameTensors = frames.map((framePath) => {
			const img = tf.node.decodeImage(fs.readFileSync(framePath));
			return tf.image.resizeBilinear(img, [384, 384]);
		});

		// Stack tensors into a single video tensor
		const videoTensor = tf.stack(frameTensors);
		const padded = tf.pad(
			videoTensor,
			[
				[0, 0],
				[0, 0],
				[0, 0],
				[0, 0],
			],
			"constant",
			0
		);
		const reshaped = padded.transpose([0, 3, 1, 2]);

		return tf.cast(reshaped, "int32");
	} catch (error) {
		console.error("Error processing video:", error);
	}
}

// Function to extract skeleton data from a video
async function skeleExec(videoPath, batchSize, detector, observation) {
	// Use ffprobe to get video metadata
	const probe = await new Promise((resolve, reject) => {
		ffmpeg.ffprobe(videoPath, (err, data) => {
			if (err) return reject(err);
			resolve(data);
		});
	});

	// Extract video frame count and process video into tensors
	const videoStreams = probe.streams.filter(
		(stream) => stream.codec_type === "video"
	);
	const frameCount = parseInt(videoStreams[0].nb_frames, 10);
	console.log(frameCount);

	// Process video frames
	const videoTensor = await processVideo(videoPath);
	console.log(videoTensor);

	const frames = await videoTensor.data();
	const normalizedFrames = frames.map((frame) => frame / 255.0);

	// Load YOLO model
	const yolo_human_model = await onnx.InferenceSession.create("yolov7.onnx");
	console.log(yolo_human_model);

	// Prepare input tensor for the YOLO model
	const batchTensor = new onnx.Tensor(
		"float32",
		new Float32Array(normalizedFrames),
		[77, 3, 384, 384]
	);
	const outputs = await yolo_human_model.run({ "onnx::Cast_0": batchTensor });

	// TODO: rewrite nonMaxSuppression function
	const output = await nonMaxSuppression(outputs["1208"]);
}

class AlphaPoseDetector {
	constructor() {
		this.poseModel = null;
		this.heatmapToCoord = heatmapToCoordSimpleRegress;
		this.evalJoints = Array.from({ length: 136 }, (_, i) => i);
	}

	// Initialize pose estimation model
	async initModel(modelPath) {
		try {
			this.poseModel = await onnx.InferenceSession.create(modelPath);
			console.log("ONNX pose model loaded successfully");
		} catch (error) {
			console.error("Failed to load ONNX model:", error);
		}
	}

	// Estimate poses from input tensor
	async getPoseEstimations(tensor) {
		try {
			let heatmaps = await this.poseModel.run({ "input.1": tensor });
			heatmaps = heatmaps["627"] || [];

			const poseEstimations = [];

			for (let i = 0; i < heatmaps.dims[2]; i++) {
				const hm = heatmaps.cpuData;
				const dummyBbox = [0, 0, 128, 128];
				const [poseCoord, poseScore] = this.heatmapToCoord(
					hm[this.evalJoints],
					dummyBbox,
					[64, 48],
					"sigmoid"
				);
				const poseEstimation = tf.concat([poseCoord, poseScore], 1).arraySync();
				poseEstimations.push(poseEstimation);
			}

			return tf.tensor2d(poseEstimations);
		} catch (error) {
			console.error("Error during pose estimation:", error);
		}
	}
}

export async function runAlphaPose() {
	// Path to AlphaPose model
	const alphaPoseModelPath = "alpha_pose.onnx";
	const detector = new AlphaPoseDetector();
	await detector.initModel(alphaPoseModelPath);

	// Path to input video
	const inputVideo = "vadim.mp4";
	skeleExec(inputVideo, batchSize, detector, "observation");
}

runAlphaPose();
