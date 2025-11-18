// Loopy Belief Propagation for Stereo Matching
//

#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

typedef vector<int> IntRow;
typedef vector<IntRow> IntTable;
typedef vector<int> Msg;
typedef vector<Msg> MsgRow;
typedef vector<MsgRow> MsgTable;

Mat leftImage, rightImage, disparityMap, disparityImage;
IntTable smoothnessCost;
MsgTable dataCost, msgUp, msgDown, msgRight, msgLeft;
int width, height, levels, iterations, lambda, truncationThreshold;

void sendMessageUp(int x, int y);
void sendMessageDown(int x, int y);
void sendMessageRight(int x, int y);
void sendMessageLeft(int x, int y);
void createMessage(Msg &msgData, Msg &msgIn1, Msg &msgIn2, Msg &msgIn3, Msg &msgOut);
int computeDataCost(int x, int y, int label);
int computeSmoothnessCost(int label1, int label2);
int findBestAssignment(int x, int y);
int computeBelief(int x, int y, int label);
int computeEnergy();

int main()
{
	levels = 16;
	iterations = 50;
	lambda = 5;
	truncationThreshold = 2;

	// Start timer
	auto start = chrono::steady_clock::now();

	// Read stereo image
	leftImage = imread("left.png", IMREAD_GRAYSCALE);
	rightImage = imread("right.png", IMREAD_GRAYSCALE);

	// Use gaussian filter
	GaussianBlur(leftImage, leftImage, Size(5, 5), 0.68);
	GaussianBlur(rightImage, rightImage, Size(5, 5), 0.68);

	// Get image size
	width = leftImage.cols;
	height = leftImage.rows;

	// Cache data cost
	dataCost = MsgTable(height, MsgRow(width, Msg(levels)));
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
			for (int i = 0; i < levels; i++)
				dataCost[y][x][i] = computeDataCost(x, y, i);

	// Cache smoothness cost
	smoothnessCost = IntTable(levels, IntRow(levels));
	for (int i = 0; i < levels; i++)
		for (int j = 0; j < levels; j++)
			smoothnessCost[i][j] = computeSmoothnessCost(i, j);

	// Initialize disparity map
	disparityMap = Mat::zeros(height, width, CV_8U);

	// Initialize messages
	msgUp = MsgTable(height, MsgRow(width, Msg(levels, 0)));
	msgDown = MsgTable(height, MsgRow(width, Msg(levels, 0)));
	msgRight = MsgTable(height, MsgRow(width, Msg(levels, 0)));
	msgLeft = MsgTable(height, MsgRow(width, Msg(levels, 0)));

	cout << "Iter.\tEnergy" << endl;
	cout << "--------------" << endl;

	// Start iterations
	for (int iter = 1; iter <= iterations; iter++)
	{
		// Update messages (Accelerated2)
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				if (x < width - 1)
					sendMessageRight(x, y);
				if (y < height - 1)
					sendMessageDown(x, y);
			}
		for (int y = height - 1; y >= 0; y--)
			for (int x = width - 1; x >= 0; x--)
			{
				if (x > 0)
					sendMessageLeft(x, y);
				if (y > 0)
					sendMessageUp(x, y);
			}

		// Update disparity map
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++)
			{
				int label = findBestAssignment(x, y);
				disparityMap.at<uchar>(y, x) = label;
			}

		// Update disparity image
		int scaleFactor = 256 / levels;
		disparityMap.convertTo(disparityImage, CV_8U, scaleFactor);

		// Show energy
		int energy = computeEnergy();
		cout << iter << "\t" << energy << endl;

		// Show disparity image
		namedWindow("Disparity Image", WINDOW_NORMAL);
		imshow("Disparity Image", disparityImage);
		waitKey(1);
	}

	// Save disparity image
	imwrite("disparity.png", disparityImage);

	// Stop timer
	auto end = chrono::steady_clock::now();
	auto diff = end - start;
	cout << "\nRunning Time: " << chrono::duration<double, milli>(diff).count() << " ms" << endl;

	waitKey(0);

	return 0;
}

void sendMessageUp(int x, int y)
{
	Msg &msgOut = msgDown[y - 1][x];
	createMessage(dataCost[y][x], msgDown[y][x], msgRight[y][x], msgLeft[y][x], msgOut);
}

void sendMessageDown(int x, int y)
{
	Msg &msgOut = msgUp[y + 1][x];
	createMessage(dataCost[y][x], msgUp[y][x], msgRight[y][x], msgLeft[y][x], msgOut);
}

void sendMessageRight(int x, int y)
{
	Msg &msgOut = msgLeft[y][x + 1];
	createMessage(dataCost[y][x], msgUp[y][x], msgDown[y][x], msgLeft[y][x], msgOut);
}

void sendMessageLeft(int x, int y)
{
	Msg &msgOut = msgRight[y][x - 1];
	createMessage(dataCost[y][x], msgUp[y][x], msgDown[y][x], msgRight[y][x], msgOut);
}

void createMessage(Msg &msgData, Msg &msgIn1, Msg &msgIn2, Msg &msgIn3, Msg &msgOut)
{
	// Create message
	for (int i = 0; i < levels; i++)
	{
		int min = INT_MAX;
		for (int j = 0; j < levels; j++)
		{
			int cost = msgData[j] + smoothnessCost[i][j]
				+ msgIn1[j] + msgIn2[j] + msgIn3[j];
			if (cost < min)
				min = cost;
		}
		msgOut[i] = min;
	}

	// Normalize message
	int min = INT_MAX;
	for (int i = 0; i < levels; i++)
		if (msgOut[i] < min)
			min = msgOut[i];
	for (int i = 0; i < levels; i++)
		msgOut[i] -= min;
}

int computeDataCost(int x, int y, int label)
{
	int leftPixel = leftImage.at<uchar>(y, x);
	int rightPixel = (x >= label) ? rightImage.at<uchar>(y, x - label) : 0;
	int cost = abs(leftPixel - rightPixel);

	return cost;
}

int computeSmoothnessCost(int label1, int label2)
{
	int cost = lambda * min(abs(label1 - label2), truncationThreshold);

	return cost;
}

int findBestAssignment(int x, int y)
{
	int label, min = INT_MAX;
	for (int i = 0; i < levels; i++)
	{
		int cost = computeBelief(x, y, i);
		if (cost < min)
		{
			label = i;
			min = cost;
		}
	}

	return label;
}

int computeBelief(int x, int y, int label)
{
	int cost = dataCost[y][x][label] + msgUp[y][x][label] + msgDown[y][x][label] + msgRight[y][x][label] + msgLeft[y][x][label];

	return cost;
}

int computeEnergy()
{
	int energy = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			int label1 = disparityMap.at<uchar>(y, x);
			energy += dataCost[y][x][label1];
			if (x < width - 1)
			{
				int label2 = disparityMap.at<uchar>(y, x + 1);
				energy += smoothnessCost[label1][label2];
			}
			if (y < height - 1)
			{
				int label3 = disparityMap.at<uchar>(y + 1, x);
				energy += smoothnessCost[label1][label3];
			}
		}

	return energy;
}
