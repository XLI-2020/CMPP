/**
 * 
 */
package indoor_entitity;


/**
 * <h>Range</h>
 * an interface Range
 *
 */
public interface Range {
	/** get minimum distance between the Range and one Point */
	public abstract double getMinDist(Point point);
	
	/** get maximum distance between the Range and one Point */
	public abstract double getMaxDist(Point point);	
}
