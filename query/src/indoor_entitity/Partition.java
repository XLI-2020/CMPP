package indoor_entitity;

import iDModel.Topology;
import utilities.DataGenConstant;

import java.io.IOException;
import java.util.*;

import iDModel.DistMatrix;
import utilities.DoorType;
import utilities.Functions;


/**
 * <h>Partition</h> to describe a partition
 *
 */
public class Partition extends Rect {
	private int mID; // the ID

	private int mType; // room = 0; hallway = 1; staircase =2;

	private int mFloor; // the floor that the partition located on

	private ArrayList<Integer> mDoors = new ArrayList<Integer>(); // the Doors of this partition

	private HashMap<String, D2Ddistance> d2dHashMap = new HashMap<String, D2Ddistance>(); // the distance between the relevant doors of this partition

	private DistMatrix distMatrix; // the distance between the doors of this partition

	private Topology topology = new Topology(); // the connectivity tier of this partition

	private HashMap<Integer, ArrayList<Double>> population_le = new HashMap<>();

	private HashMap<Integer, ArrayList<Double>> population_ge = new HashMap<>();

	private HashMap<Integer, ArrayList<Double>> population_stgcn = new HashMap<>();

	private HashMap<Integer, ArrayList<Double>> population_tgcn = new HashMap<>();

	private HashMap<Integer, ArrayList<Double>> population_astgnn = new HashMap<>();

	private double area;

	private int tType;


	/**
	 * Constructor
	 * 
	 * @param x1
	 * @param x2
	 * @param y1
	 * @param y2
	 * @param mType
	 */
	public Partition(double x1, double x2, double y1, double y2, int mType) {
		super(x1, x2, y1, y2);
		this.mFloor = DataGenConstant.mID_Floor;
		this.mID = DataGenConstant.mID_Par++;
		this.mType = mType;
		calArea();
//		System.out.println("partition generated with id = " + this.mID + " type = " + this.mType + " on floor = " + this.mFloor);
	}

	/**
	 * Constructor
	 * 
	 * @param another
	 */
	public Partition(Partition another) {
		super(another.getX1(), another.getX2(), another.getY1(), another.getY2());
		this.setmType(another.getmType());
		this.setmFloor(another.getmFloor());
		calArea();
	}

	/**
	 * add a relevant door of this partition
	 * 
	 * @param doorID
	 */
	public void addDoor(int doorID) {
		if (!this.mDoors.contains(doorID)) {
			this.mDoors.add(doorID);
		}
	}


	public void addPopulation_le(int t, double a, double u) {
		ArrayList<Double> temp = new ArrayList<>(Arrays.asList(a, u));
		this.population_le.put(t, temp);
	}

	public HashMap<Integer, ArrayList<Double>> getPopulation_le() {
		return this.population_le;
	}

	public ArrayList<Double> getPopulation_le(int t) {
		return this.population_le.get(t);
	}

	public void addPopulation_ge(int t, double a, double u) {
		ArrayList<Double> temp = new ArrayList<>(Arrays.asList(a, u));
		this.population_ge.put(t, temp);
	}

	public HashMap<Integer, ArrayList<Double>> getPopulation_ge() {
		return this.population_ge;
	}

	public ArrayList<Double> getPopulation_ge(int t) {
		return this.population_ge.get(t);
	}


	public void addPopulation_stgcn(int t, double a, double u) {
		ArrayList<Double> temp = new ArrayList<>(Arrays.asList(a, u));
		this.population_stgcn.put(t, temp);
	}

	public HashMap<Integer, ArrayList<Double>> getPopulation_stgcn() {
		return this.population_stgcn;
	}

	public ArrayList<Double> getPopulation_stgcn(int t) {
		return this.population_stgcn.get(t);
	}


	public void addPopulation_tgcn(int t, double a, double u) {
		ArrayList<Double> temp = new ArrayList<>(Arrays.asList(a, u));
		this.population_tgcn.put(t, temp);
	}

	public HashMap<Integer, ArrayList<Double>> getPopulation_tgcn() {
		return this.population_tgcn;
	}

	public ArrayList<Double> getPopulation_tgcn(int t) {
		return this.population_tgcn.get(t);
	}

	public void addPopulation_astgnn(int t, double a, double u) {
		ArrayList<Double> temp = new ArrayList<>(Arrays.asList(a, u));
		this.population_astgnn.put(t, temp);
	}

	public HashMap<Integer, ArrayList<Double>> getPopulation_astgnn() {
		return this.population_astgnn;
	}

	public ArrayList<Double> getPopulation_astgnn(int t) {
		return this.population_astgnn.get(t);
	}



	/**
	 * @return the mType
	 */
	public int getmType() {
		return mType;
	}

	/**
	 * @param mType the mType to set
	 */
	public void setmType(int mType) {
		this.mType = mType;
	}

	/**
	 * @return the tType
	 */
	public int gettType() {
		return tType;
	}

	/**
	 * @param tType the tType to set
	 */
	public void settType(int tType) {
		this.tType = tType;
	}


	/**
	 * @return the mFloor
	 */
	public int getmFloor() {
		return mFloor;
	}

	/**
	 * @param mFloor the mFloor to set
	 */
	public void setmFloor(int mFloor) {
		this.mFloor = mFloor;
	}

	/**
	 * @return the mDoors
	 */
	public ArrayList<Integer> getmDoors() {
		return mDoors;
	}

	/**
	 * @param mDoors the mDoors to set
	 */
	public void setmDoors(ArrayList<Integer> mDoors) {
		this.mDoors = mDoors;
	}

	/**
	 * @return the mID
	 */
	public int getmID() {
		return mID;
	}

	/**
	 * @param mID the mID to set
	 */
	public void setmID(int mID) {
		this.mID = mID;
	}

	/**
	 * @return the distMatrix
	 */
	public DistMatrix getdistMatrix() {
		return distMatrix;
	}

	/**
	 * @param distMatrix the distMatrix to set
	 */
	public void setDistMatrix(DistMatrix distMatrix) {
		this.distMatrix = distMatrix;
//		System.out.println(Functions.print2Ddoublearray(this.distMatrix.getMatrix(), 0, this.distMatrix.getMatrix().length
//				, 0, this.distMatrix.getMatrix()[0].length));
	}

	/**
	 * @return the topology
	 */
	public Topology getTopology() {
		return topology;
	}

	/**
	 * @param topology the topology to set
	 */
	public void setTopology(Topology topology) {
		this.topology = topology;
	}

	/**
	 * @return the shop
	 */
	public Shop getshop() {
		return null;
	}

	/**
	 * @return the d2dHashMap
	 */
	public HashMap<String, D2Ddistance> getD2dHashMap() throws IOException {
		d2dHashMap = new HashMap<String, D2Ddistance>();
		Collections.sort(this.mDoors);

		int doorSize = this.mDoors.size();
		for (int i = 0; i < doorSize; i++) {
			int index_1 = this.mDoors.get(i);
			Door door1 = IndoorSpace.iDoors.get(index_1);

			int doorSize1 = this.mDoors.size();
			for (int j = i + 1; j < doorSize1; j++) {
				int index_2 = this.mDoors.get(j);
				Door door2 = IndoorSpace.iDoors.get(index_2);
				D2Ddistance d2dDist;
				if (door2.getmType() == DoorType.EXIT || door1.getmType() == DoorType.EXIT) {
					d2dDist = new D2Ddistance(index_1, index_2, DataGenConstant.lenStairway);
				}
				else {
//					if (DataGenConstant.divisionType == 0 && this.getmType() == RoomType.HALLWAY) {
////						double dist = 0;
////						Path path = Paths.get(System.getProperty("user.dir") + "/hallway_distMatrix_division_0" + ".txt");
////						Scanner scanner = new Scanner(path);
////
////						//read line by line
////						Boolean flag = false;
////						while(scanner.hasNextLine()){
////							//process each line
////							String line = scanner.nextLine();
////							String[] tempArr = line.split("\t");
////							if ((door1.getX() + "." + door1.getY() + "-" + door2.getX() + "." + door2.getY()).equals(tempArr[1])) {
////								dist = Double.parseDouble(tempArr[2]);
////								flag = true;
////								break;
////							}
////						}
////						if (!flag) {
////							System.out.println("something wrong with hallway distance matrix");
////							System.out.println(door1.getX() + "." + door1.getY() + "-" + door2.getX() + "." + door2.getY());
////						}
////
////						d2dDist = new D2Ddistance(index_1, index_2, dist);
//					}
//					else {
//						d2dDist = new D2Ddistance(index_1, index_2, door1.eDist(door2));
//					}
					d2dDist = new D2Ddistance(index_1, index_2, door1.eDist(door2));
				}

				d2dHashMap.put(Functions.keyConventer(index_1, index_2), d2dDist);
			}
		}
		this.setD2dHashMap(d2dHashMap);

		return d2dHashMap;
	}

	/**
	 * calculate area of this partition
	 */
	public void calArea() {
		double area = Math.abs(this.getX2() - this.getX1()) * Math.abs(this.getY2() - this.getY1());
		this.area = area;
	}

	/**
	 * set the area of this partiton
	 * @param area
	 */
	public void setArea(double area) {
		this.area = area;
	}

	/**
	 * get the area of this partition
	 * @return
	 */
	public double getArea() {
		return this.area;
	}

	/**
	 * @param d2dHashMap the d2dHashMap to set
	 */
	public void setD2dHashMap(HashMap<String, D2Ddistance> d2dHashMap) {
		this.d2dHashMap = d2dHashMap;
	}

	/**
	 * toString
	 * 
	 * @return mID+x1+x2+y1+y2+mFloor+mType+mDoors
	 */
	public String toString() {
		String outputString = this.getmID() + "\t" + this.getX1() + "\t" + this.getX2() + "\t" + this.getY1() + "\t"
				+ this.getY2() + "\t" + this.getmFloor() + "\t" + this.getmType() + "\t";

		Iterator<Integer> itr = this.mDoors.iterator();

		while (itr.hasNext()) {
			outputString = outputString + itr.next() + "\t";
		}

		return outputString;
	}

	public String cornerToString3D() {
		return "(" + this.getX1() + ", " + this.getY1() + ", " + this.getmFloor() + ")" + ", (" + this.getX2() + ", "
				+ this.getY2() + ", " + (this.getmFloor() + 1) + ")";
	}

	public String cornerToString2D() {
		return "(" + this.getX1() + ", " + this.getY1() + ")" + ", (" + this.getX2() + ", " + this.getY2() + ")";
	}

	/**
	 * d2DtoString
	 * 
	 * @return d11 d12 d13 \n d21 d22 ... dnn
	 */
	public String d2DtoString() {
		String outputString = this.mID + "\t";

		Iterator it = this.d2dHashMap.entrySet().iterator();

		while (it.hasNext()) {
			Map.Entry pair = (Map.Entry) it.next();
			outputString = outputString + pair.getKey() + " = " + pair.getValue() + "\t";
		}

		return outputString;
	}

	/**
	 * @param doorID
	 * @return the index of the door in mDoors list
	 */
	public int getDoorIndex(int doorID) {
		int result = -1;

		int doorSize = this.mDoors.size();
		for (int i = 0; i < doorSize; i++) {
			if (doorID == this.mDoors.get(i)) {
				result = i;
				break;
			}
		}

		return result;
	}

	/**
	 * test this LeavePair can be fully covered during the walking period
	 *
	 * @param remainDist
	 * @return a boolean value
	 */
	public double canBeFullyCovered(Door curDoor, double remainDist) {
		// TODO Auto-generated method stub

		double maxDist = this.getMaxDist(curDoor);

		if(maxDist <= remainDist){
			return remainDist - maxDist;
		}else
			return -1;
	}

	// check whether a point is in this partition
	public boolean isInPartition(Point point) {
		double x1 = this.getX1();
		double y1 = this.getY1();
		double z1 = this.mFloor;
		double x2 = this.getX2();
		double y2 = this.getY2();
		double z2 = this.mFloor;

		double x = point.getX();
		double y = point.getY();
		double z = point.getmFloor();

		if (x1 <= x && x <= x2) {
			if (y1 <= y && y <= y2) {
				if (z1 <= z && z <= z2) {
					return true;
				} else
					return false;
			} else
				return false;
		} else
			return false;
	}

	public void setshop(Shop shop2) {
		// TODO Auto-generated method stub
		
	}

	public int[] getmKeywords() {
		// TODO Auto-generated method stub
		return null;
	}

	public int getIkeyword() {
		// TODO Auto-generated method stub
		return 0;
	}


}
