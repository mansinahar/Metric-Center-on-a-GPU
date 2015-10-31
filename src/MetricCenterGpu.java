import java.io.BufferedReader;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuDoubleArray;
import edu.rit.gpu.GpuStructArray;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Module;
import edu.rit.gpu.Struct;
import edu.rit.pj2.Task;


public class MetricCenterGpu extends Task {
	
	private static class Point extends Struct
	{
		public double x;
		public double y;
		
		public Point(double x, double y)
		{
			this.x = x;
			this.y = y;
		}
		
		public static long sizeof()
		{
			return 16;
		}

		@Override
		public void toStruct(ByteBuffer buf) {
			buf.putDouble(x);
			buf.putDouble(y);
		}

		@Override
		public void fromStruct(ByteBuffer buf) {
			x = buf.getDouble();
			y = buf.getDouble();
			
		}
		
	}
	
	// Interface for the kernel
	private static interface MetricCenterKernel extends Kernel {
		public void metricCenter(GpuStructArray<Point> allPoints, int n);
	}

	double[] xPoints;
	double[] yPoints;
	int n;
	GpuStructArray<Point> allPoints;
	GpuDoubleArray result;
	
	
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
		n = allXPoints.size();
		xPoints = new double[n];
		yPoints = new double[n];
		
		for(int i = 0; i < n; ++i)
		{
			xPoints[i] = allXPoints.get(i);
			yPoints[i] = allYPoints.get(i);
		}
		
		
	}
	
	
	
	
	public void main(String args[]) throws Exception {
		
		// Read all points from the array
		readFile(args[0]);
	
		int noOfBlocks = 3;
		
		for(int i = 0; i < xPoints.length; ++i) {
			System.out.println(xPoints[i] + " " + yPoints[i]);
		}
		
		// Get GPU object
		Gpu gpu = Gpu.gpu();
		gpu.ensureComputeCapability (2, 0);
		
		// Get the kernel module
		Module module = gpu.getModule("MetricCenterGpu.cubin");
		
		allPoints = gpu.getStructArray(Point.class, n);
		result = module.getDoubleArray("devResult", noOfBlocks);
		
		for(int i = 0; i < n; ++i)
		{
			allPoints.item[i] = new Point(xPoints[i], yPoints[i]);
		}
		
		// Get the kernel
		MetricCenterKernel kernel = module.getKernel(MetricCenterKernel.class);
		// Setup the GPU grid
		kernel.setBlockDim (256);
		kernel.setGridDim (noOfBlocks);
		
		// Copy array of points from CPU to GPU
		allPoints.hostToDev();
		
		for(int i = 0; i < noOfBlocks; ++i)
		{
			result.item[i] = -100.00;
		}
		
		// Copy result array from CPU to GPU
		result.hostToDev();
		
		// execute the kernel function
		kernel.metricCenter(allPoints, n);
		
		// Copy results back to the CPU from GPU
		result.devToHost();
		
		// Print results
		for(int i = 0; i < noOfBlocks; ++i)
		{
			System.out.println("Result is: " + result.item[i]);
		}
	}
	
	protected static int gpusRequired()
	{
		return 1;
	}
	
	protected static int coresRequired()
	{
		return 1;
	}
	
}
