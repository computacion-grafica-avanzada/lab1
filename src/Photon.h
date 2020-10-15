#ifndef __PHOTON_H
#define __PHOTON_H

#include "gdt/math/vec.h"
#include <iostream>

using namespace gdt;

/**
 * Photon storable in photon map.
 */

class Photon {
	friend class PhotonMap; // doe to efficiency ;-)
	
    protected:
	static const double COS_THETA [256];
	static const double SIN_THETA [256];
	static const double COS_PHI   [256];
	static const double SIN_PHI   [256];
	
	float			_position[3];
	vec3f			_power;
	unsigned char	_phi;   //!< azimuth angle with respect to Z coord !!!
	unsigned char	_theta; //!< elevation angle with respect to Z coord !!!
	short			_plane;
	vec3f			_direction;

    public:
	/**
	 * Implicit constructor sets all values to zero
	 */
	Photon ();

	/**
	 * Constructor.
	 *
	 * @param pos Photon's location
	 * @param dir Photon's normalized directon
	 * @param pow Phoron's power (color)
	 */
	Photon (const vec3f& pos, const vec3f& dir, const vec3f& pow);

	/**
	 * Cloning constructor
	 *
	 * @param src Source photon
	 */
	Photon (const Photon& src);
	
	Photon& operator =(const Photon&);

	virtual ~Photon () {}

	/**
	 * @return Photon's direction in world coordinates
	 */
	vec3f getDirection () const {
	    /*return vec3f(
			SIN_THETA[_theta] * COS_PHI[_phi],
			SIN_THETA[_theta] * SIN_PHI[_phi],
			COS_THETA[_theta]
		);*/
		return _direction;
	}

	/**
	 * @return Photon's location in the world coordinates
	 */
	vec3f getLocation () const { return vec3f(_position[0],
						      _position[1],
						      _position[2]); }

	/**
	 * @return Photon's power
	 */
	vec3f getPower() const { return _power; }

	/**
	 * Sets nre location.
	 *
	 * @param loc New location
	 */
	void setLocation(const vec3f& loc) {
	    _position[0] = loc.x;
	    _position[1] = loc.y;
	    _position[2] = loc.z;
	}

	/**
	 * @param out ostream
	 * @param photon photon to print out
	 * @return reference to changed output stream
	 */
	friend std::ostream& operator <<(std::ostream& out, const Photon& photon) {
		out << "Location:  " << photon.getLocation().x << " " << photon.getLocation().y << " " << photon.getLocation().z << std::endl;
		out << "Direction: " << photon.getDirection().x << " " << photon.getDirection().y << " " << photon.getDirection().z << " " << std::endl;
		out << "Power:     " << photon._power.x << " " << photon._power.y << " " << photon._power.z << std::endl;
		out << "Plane:     " << photon._plane << std::endl;
	    return out;
	}

};

#endif // __PHOTON_H
