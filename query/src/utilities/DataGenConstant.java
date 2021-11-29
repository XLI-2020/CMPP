/**
 * 
 */
package utilities;

import java.util.ArrayList;
import java.util.Arrays;

public class DataGenConstant {
	public static String dataset = "syn";
	
	// PARAMETERS FOR INDOOR SPACES
	/** dimensions of the floor */
	public static double floorRangeX = 1368;
	public static double floorRangeY = 1368;

	public static double zoomLevel = 0.6;

	/** numbers of the floor */
	public static int nFloor = 5;

	/** type of dataset */
	public static int dataType = 1; // 1 means regular dataset; 0 means less doors; 2 means more doors;

	/** type of division */
	public static int divisionType = 1; // 1 means regular division; 0 means no division for hallway; 2 means more hallway;

	/** length of stairway between two floors */
	public static double lenStairway = 20.0;

	public static int startTime = 0;

	public static int endTime = 3600;

	public static int objects = 600;

	public static ArrayList<Integer> exitDoors = new ArrayList<>(Arrays.asList(210, 212, 213, 214));

	public static int destributionPara = 10;

	public static int volatilityPara = 10;

	public static int volatilityTheta = 3;
	
	// ID COUNTERS FOR INDOOR ENTITIES
	/** the ID counter of Partitions */
	public static int mID_Par = 0;

	/** the ID counter of Doors */
	public static int mID_Door = 0;
	
	/** the ID counter of Floors */
	public static int mID_Floor = 0;

	/** the ID counter of Objects */
	public static int mID_Object = 0;
	
	// KEYWORDS	
	public static int mKeyworSize = 0;

	// traveling speed 83.34m/min
	public static double traveling_speed = 1 * 83.34 / 60;

//	public static int safe_duration = 60;

	public static void init(String dataName) {
		if (dataName.equals("hsm")) {
			dataset = "hsm";
			floorRangeX = 2100;
			floorRangeY = 2700;
			zoomLevel = 0.28;
			nFloor = 7;
//			parInterval = 5;
//			doorInterval = new ArrayList<>(Arrays.asList(5, 10, 15, 20, 25, 30, 35, 45, 50));
			objects = 600;
//
			exitDoors = new ArrayList<>(Arrays.asList(30, 37, 39, 40, 41, 43, 44, 48, 51, 52, 53, 329, 336, 338, 339, 340, 342, 343, 347, 350, 351, 352, 628, 635, 637, 638, 639, 641, 642, 646, 649, 650, 651, 927, 934, 936, 937, 938, 940, 941, 945, 948, 949, 950, 1226, 1233, 1235, 1236, 1237, 1239, 1240, 1244, 1247, 1248, 1249, 1525, 1532, 1534, 1535, 1536, 1538, 1539, 1543, 1546, 1547, 1548, 1824, 1831, 1833, 1834, 1835, 1837, 1838, 1842, 1845, 1846, 1847));
			destributionPara = 10;
			volatilityPara = 10;
			volatilityTheta = 3;


		}

		if (dataName.equals("newHsm")) {
			dataset = "newHsm";
			floorRangeX = 2100;
			floorRangeY = 2700;
			zoomLevel = 0.28;
			nFloor = 7;
			objects = 10;

			exitDoors = new ArrayList<>(Arrays.asList(1, 5, 9, 10, 233, 617, 619, 620, 622, 626));
			destributionPara = 10;
			volatilityPara = 10;
			volatilityTheta = 3;


		}

	}

}
