import * as tf from "@tensorflow/tfjs-node";

export function heatmapToCoordSimpleRegress(
	preds,
	bbox,
	hmShape,
	normType,
	hmsFlip = null
) {
	const integralOp = (hm1d) => {
		const length = hm1d.shape[hm1d.shape.length - 1];
		const range = tf.range(0, length, 1, "float32");
		return tf.mul(hm1d, range);
	};

  console.log(preds)

	if (preds.rank === 3) {
		preds = preds.expandDims(0);
	}
	const [hmHeight, hmWidth] = hmShape;
	const numJoints = preds.shape[1];

	const { predJts, predScores } = integralTensor(
		preds,
		numJoints,
		false,
		hmWidth,
		hmHeight,
		1,
		integralOp,
		normType
	);
	const reshapedPredJts = predJts.reshape([predJts.shape[0], numJoints, 2]);

	if (hmsFlip) {
		if (hmsFlip.rank === 3) {
			hmsFlip = hmsFlip.expandDims(0);
		}
		const { predJtsFlip, predScoresFlip } = integralTensor(
			hmsFlip,
			numJoints,
			false,
			hmWidth,
			hmHeight,
			1,
			integralOp,
			normType
		);
		const reshapedPredJtsFlip = predJtsFlip.reshape([
			predJtsFlip.shape[0],
			numJoints,
			2,
		]);

		predsJts = tf.add(reshapedPredJts, reshapedPredJtsFlip).div(2);
		predScores = tf.add(predScores, predScoresFlip).div(2);
	}

	const coords = reshapedPredJts
		.arraySync()
		.map((row) =>
			row.map(([x, y]) => [(x + 0.5) * hmWidth, (y + 0.5) * hmHeight])
		);

	const xmin = bbox[0],
		ymin = bbox[1],
		xmax = bbox[2],
		ymax = bbox[3];
	const w = xmax - xmin;
	const h = ymax - ymin;
	const center = [xmin + w * 0.5, ymin + h * 0.5];
	const scale = [w, h];

	preds = coords.map((row) =>
		row.map(([x, y]) =>
			transformPreds([x, y], center, scale, [hmWidth, hmHeight])
		)
	);

	const outputPreds = preds.length === 1 ? preds[0] : preds;
	const outputScores = predScores.arraySync();

	return [outputPreds, outputScores];
}

function transformPreds(coords, center, scale, outputSize) {
	const targetCoords = [...coords];
	const trans = getAffineTransform(center, scale, 0, outputSize, true);
	const transformedCoords = affineTransform(coords.slice(0, 2), trans);
	targetCoords[0] = transformedCoords[0];
	targetCoords[1] = transformedCoords[1];
	return targetCoords;
}

function affineTransform(pt, t) {
	const newPt = [pt[0], pt[1], 1.0];
	const transformed = t.map((row) =>
		row.reduce((sum, val, idx) => sum + val * newPt[idx], 0)
	);
	return transformed.slice(0, 2);
}

function getAffineTransform(
	center,
	scale,
	rot,
	outputSize,
	shift = [0, 0],
	inv = false
) {
	if (!Array.isArray(scale)) {
		scale = [scale, scale];
	}

	const srcW = scale[0];
	const dstW = outputSize[0];
	const dstH = outputSize[1];

	const rotRad = (Math.PI * rot) / 180;
	const srcDir = getDir([0, -0.5 * srcW], rotRad);
	const dstDir = [0, -0.5 * dstW];

	const src = [
		[center[0] + scale[0] * shift[0], center[1] + scale[1] * shift[1]],
		[
			center[0] + srcDir[0] + scale[0] * shift[0],
			center[1] + srcDir[1] + scale[1] * shift[1],
		],
		get3rdPoint(
			[center[0] + scale[0] * shift[0], center[1] + scale[1] * shift[1]],
			[
				center[0] + srcDir[0] + scale[0] * shift[0],
				center[1] + srcDir[1] + scale[1] * shift[1],
			]
		),
	];

	const dst = [
		[dstW * 0.5, dstH * 0.5],
		[dstW * 0.5 + dstDir[0], dstH * 0.5 + dstDir[1]],
		get3rdPoint(
			[dstW * 0.5, dstH * 0.5],
			[dstW * 0.5 + dstDir[0], dstH * 0.5 + dstDir[1]]
		),
	];

	const srcMat = cv.matFromArray(3, 2, cv.CV_32F, src.flat());
	const dstMat = cv.matFromArray(3, 2, cv.CV_32F, dst.flat());

	let trans;
	if (inv) {
		trans = cv.getAffineTransform(dstMat, srcMat);
	} else {
		trans = cv.getAffineTransform(srcMat, dstMat);
	}

	srcMat.delete();
	dstMat.delete();

	return trans;
}

function get3rdPoint(a, b) {
	const direct = [a[0] - b[0], a[1] - b[1]];
	return [b[0] - direct[1], b[1] + direct[0]];
}

function getDir(srcPoint, rotRad) {
	const sin = Math.sin(rotRad);
	const cos = Math.cos(rotRad);

	return [
		srcPoint[0] * cos - srcPoint[1] * sin,
		srcPoint[0] * sin + srcPoint[1] * cos,
	];
}

function integralTensor(
	preds,
	numJoints,
	output3D,
	hmWidth,
	hmHeight,
	hmDepth,
	integralOperation,
	normType = "softmax"
) {
	preds = preds.reshape([preds.shape[0], numJoints, -1]);
	preds = normHeatmap(normType, preds);

	let maxvals;
	if (normType === "sigmoid") {
		maxvals = tf.max(preds, 2, true);
	} else {
		maxvals = tf.ones([preds.shape[0], numJoints, 1], "float32");
	}

	const heatmaps = preds
		.div(preds.sum(2, true))
		.reshape([preds.shape[0], numJoints, hmDepth, hmHeight, hmWidth]);

	const hmX = heatmaps.sum([2, 3]);
	const hmY = heatmaps.sum([2, 4]);
	const hmZ = heatmaps.sum([3, 4]);

	const coordX = integralOperation(hmX).sum(2, true).div(hmWidth).sub(0.5);
	const coordY = integralOperation(hmY).sum(2, true).div(hmHeight).sub(0.5);

	let predJts;
	if (output3D) {
		const coordZ = integralOperation(hmZ).sum(2, true).div(hmDepth).sub(0.5);
		predJts = tf.concat([coordX, coordY, coordZ], 2);
		predJts = predJts.reshape([predJts.shape[0], numJoints * 3]);
	} else {
		predJts = tf.concat([coordX, coordY], 2);
		predJts = predJts.reshape([predJts.shape[0], numJoints * 2]);
	}

	return { predJts, maxvals: maxvals.toFloat() };
}

function normHeatmap(normType, heatmap) {
	const shape = heatmap.shape;
	if (normType === "softmax") {
		const reshaped = heatmap.reshape([shape[0], shape[1], -1]);
		const softmaxed = tf.softmax(reshaped, -1);
		return softmaxed.reshape(shape);
	}
	console.log("norm heat map not supported");
	return heatmap;
}
