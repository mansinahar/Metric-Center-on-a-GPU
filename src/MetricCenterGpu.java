/*
 * File: MetricCenterGpu.java
 * Package: ---
 * @author MansiNahar
 * @version 1.2
 *  
 *  Given a set of points, this program finds the metric center of those points.
 */
import java.io.BufferedReader;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuStructArray;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Module;
import edu.rit.gpu.Struct;
import edu.rit.pj2.Task;

/*
 * This class the sets up the GPU and calls the kernel function
 * to calculate the metric center of the given points
 */
public class MetricCenterGpu extends Task {
	
	// Structure for holding the result of a metric center
	private static class Result extends Struct
	{
		double distance;
		int pointIndex;
		
		/*
		 * Constructor to set the distance and pointIndex of the result
		 * @param distance
		 * 		Maximum radius needed to draw a circle
		 * 		from this point that encloses all the given points.
		 * @param pointIndex
		 * 		The point whose maximum radius is calculated such that
		 * 		the radius encloses all the given points.
		 */
		public Result(double distance, int pointIndex)
		{
			this.distance = distance;
			this.pointIndex = pointIndex;
		}
		
		/*
		 * Returns the size in bytes of the corresponding C struct.
		 * @return long
		 * 			Size in bytes of the struct.
		 */
		public static long sizeof()
		{
			return 16;
		}

		/*
		 * Write this Result object to the given byte buffer as a C struct.
		 * @param buf
		 * 			The buffer to which object needs to be written.
		 */
		public void toStruct(ByteBuffer buf) {
			buf.putDouble(distance);
			buf.putInt(pointIndex);
		}

		/*
		 * Read the C struct from the given byte buffer as a Result object.
		 * @param buf
		 * 			The buffer from which object needs to be read.
		 */
		@Override
		public void fromStruct(ByteBuffer buf) {
			distance = buf.getDouble();
			pointIndex = buf.getInt();
		}
		
	}
	
	// Structure for holding the 2D points
	private static class Point extends Struct
	{
		public double x;
		public double y;
		
		/*
		 * Constructor the set the x and y coordinates of the point.
		 * @param x
		 * 		X coordinate of the point
		 * @param y
		 * 		Y coordinate of the point
		 */
		public Point(double x, double y)
		{
			this.x = x;
			this.y = y;
		}
		
		/*
		 * Returns the size in bytes of the corresponding C struct.
		 * @return long
		 * 			Size in bytes of the struct.
		 */
		public static long sizeof()
		{
			return 16;
		}

		/*
		 * Write this Result object to the given byte buffer as a C struct.
		 * @param buf
		 * 			The buffer to which object needs to be written.
		 */
		public void toStruct(ByteBuffer buf) {
			buf.putDouble(x);
			buf.putDouble(y);
		}

		/*
		 * Read the C struct from the given byte buffer as a Result object.
		 * @param buf
		 * 			The buffer from which object needs to be read.
		 */
		public void fromStruct(ByteBuffer buf) {
			x = buf.getDouble();
			y = buf.getDouble();
		}
	}
	
	/*
	 * Interface for the kernel function that calculates the metric center
	 */
	private static interface MetricCenterKernel extends Kernel {
		
		/*
		 * Function to calculate the metric center
		 * @param allPoints
		 * 		A GpuStructArray of Point consisting of all 2D points.
		 * @param n
		 * 		Number of 2D points.
		 */
		public void metricCenter(GpuStructArray<Point> allPoints, GpuStructArray<Result> devResult, int n);
	}

	double[] xPoints;
	double[] yPoints;
	
	
	/*
	 * This method reads a file consisting of a set of x and y points
	 * @param fileName
	 * 			Name of the file from which these points need to be read.
	 * @exception Exception
	 * 			Throws general exception, if occurred 
	 */
	public void readFile(String fileName) throws Exception {
		
		ArrayList<Double> allXPoints = new ArrayList<Double>();
		ArrayList<Double> allYPoints = new ArrayList<Double>();
		
		Path file = Paths.get(fileName);
		
		// Read all points in two array list (different for x coords and y coords)
		try (BufferedReader reader = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
			String line;
			while((line = reader.readLine()) != null) {
				String[] Coords = line.split(" ");
				Double xPoint = Double.parseDouble(Coords[0]);
				Double yPoint =	Double.parseDouble(Coords[1]);
				allXPoints.add(xPoint);
				allYPoints.add(yPoint);
			}
		}
		
		
		// Copy points from array lists to arrays
		xPoints = new double[allXPoints.size()];
		yPoints = new double[allXPoints.size()];
		
		for(int i = 0; i < allXPoints.size(); ++i)
		{
			xPoints[i] = allXPoints.get(i);
			yPoints[i] = allYPoints.get(i);
		}
	}
	
	/*
	 * Reduces the GPU result
	 * @param result
	 * 			GpuStructArray of results of each block of the GPU.
	 * @param n
	 * 			Length of the result array.
	 * @return Result
	 * 			The final result after reduction
	 */
	public Result reduceResult(GpuStructArray<Result> result, int n)
	{
		Result finalResult = new Result(-100.00, -1);
		for(int i = 0; i < n; ++i)
		{
			if((finalResult.distance == -100.00 && finalResult.pointIndex == -1) || finalResult.distance > result.item[i].distance)
			{
				finalResult.distance = result.item[i].distance;
				finalResult.pointIndex = result.item[i].pointIndex;
			}
		}
		return finalResult;
	}
	
	/*
	 * Prints the Result object
	 * @param finalResult
	 * 		Result object that needs to be printed
	 * @param allPoints
	 * 		GpuStructArray of all points
	 */
	public void printResult(Result finalResult, GpuStructArray<Point> allPoints)
	{
		System.out.printf(finalResult.pointIndex + " (" + "%.5g" + "," + "%.5g" + ")", allPoints.item[finalResult.pointIndex].x, allPoints.item[finalResult.pointIndex].y);
		System.out.println();
		System.out.printf("%.5g", finalResult.distance);
		System.out.println();
	}
	
	/*
	 * Specifies the number of GPUs required by this program
	 * @return int
	 * 		Integer specifying the number of GPUs required.
	 */
	protected static int gpusRequired()
	{
		return 1;
	}
	
	/*
	 * Specifies the number of cores required by this program
	 * @return int
	 * 		Integer specifying the number of cores required.
	 */
	protected static int coresRequired()
	{
		return 1;
	}
	
	/*
	 * This function prints the usage of this program
	 */
	public void usage() {
		System.out.println("Usage:");
		System.out.println("java pj2 MetricCenterGpu <file>");
		System.out.println("file = Text file from which points need to be read.");
		throw new IllegalArgumentException();
	}
	
	/*
	 * Main function - reading all the points and calculating the metric center
	 * @param args
	 * 		File name from which points need to be read
	 * @exception Exception
	 * 		Throws general exception, if occurred.
	 */
	public void main(String args[]) throws Exception {
		
		String fileName="";
		
		// Check for the file name argument. If no such argument exists
		// or more than one argument exists then print usage.
		if(args.length != 1) {
			usage();
		}
		else {
			fileName = args[0];
		}
		
		// Read all points from the file
		readFile(fileName);
		
		int n;
		GpuStructArray<Point> allPoints;
		GpuStructArray<Result> result;
		Result finalResult = new Result(-100.00, -1);
		
		n = xPoints.length;
		
		// Get GPU object
		Gpu gpu = Gpu.gpu();
		gpu.ensureComputeCapability (2, 0);
		
		// Get the kernel module
		Module module = gpu.getModule("MetricCenterGpu.cubin");
		
		allPoints = gpu.getStructArray(Point.class, n);
		result = gpu.getStructArray(Result.class, gpu.getMultiprocessorCount());
		
		for(int i = 0; i < n; ++i)
		{
			allPoints.item[i] = new Point(xPoints[i], yPoints[i]);
		}
		
		// Get the kernel
		MetricCenterKernel kernel = module.getKernel(MetricCenterKernel.class);
		
		// Setup the GPU grid
		kernel.setBlockDim (1024);
		kernel.setGridDim (gpu.getMultiprocessorCount());
		
		// Copy array of points from CPU to GPU
		allPoints.hostToDev();
		
		for(int i = 0; i < result.length(); ++i)
		{
			result.item[i] =  new Result(-100.00, -1);
		}
		
		// Copy result array from CPU to GPU
		result.hostToDev();
		
		// execute the kernel function
		kernel.metricCenter(allPoints, result, n);
		
		// Copy results back to the CPU from GPU
		result.devToHost();
		
		// Reduce result and print it
		finalResult = reduceResult(result, result.length());
		printResult(finalResult, allPoints);
		
	}
	
}
